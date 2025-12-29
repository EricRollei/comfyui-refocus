"""
FLUX Generation utilities for Genfocus.

This module provides the core generation logic ported from Genfocus,
adapted to work with ComfyUI's model system.

Based on: https://github.com/rayray9999/Genfocus/blob/main/Genfocus/pipeline/flux.py
License: Apache 2.0
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Union, Optional, Dict, Any, Tuple
from contextlib import contextmanager
from PIL import Image
import math


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)


def clip_hidden_states(hidden_states: torch.FloatTensor) -> torch.FloatTensor:
    """Clip hidden states to prevent overflow in fp16."""
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    return hidden_states


class Condition:
    """
    Represents a conditioning image for the Genfocus pipeline.
    
    Each condition has:
    - condition: The image (PIL or tensor)
    - adapter: Which LoRA adapter to use ("deblurring" or "bokeh")
    - position_delta: Offset for position embeddings [y_offset, x_offset]
    - position_scale: Scale factor for position embeddings
    """
    
    def __init__(
        self,
        condition: Union[Image.Image, torch.Tensor],
        adapter_setting: str,
        position_delta: Optional[List[int]] = None,
        position_scale: float = 1.0,
        no_preprocess: bool = False,
    ) -> None:
        self.condition = condition
        self.adapter = adapter_setting
        self.position_delta = position_delta if position_delta is not None else [0, 0]
        self.position_scale = position_scale
        self.no_preprocess = no_preprocess


def encode_image_to_latents(vae, image: Union[Image.Image, torch.Tensor], device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode an image to FLUX latent space.
    
    Returns:
        tokens: Packed latent tokens [B, H*W/4, C*4]
        ids: Position IDs [H*W/4, 3]
    """
    # Convert PIL to tensor if needed
    if isinstance(image, Image.Image):
        img_np = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    else:
        img_tensor = image
    
    img_tensor = img_tensor.to(device=device, dtype=dtype)
    
    # Ensure BCHW format
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    if img_tensor.shape[-1] == 3:  # BHWC -> BCHW
        img_tensor = img_tensor.permute(0, 3, 1, 2)
    
    # Encode with VAE
    with torch.no_grad():
        latents = vae.encode(img_tensor)
        if hasattr(latents, 'latent_dist'):
            latents = latents.latent_dist.sample()
        elif hasattr(latents, 'sample'):
            latents = latents.sample()
        else:
            # ComfyUI VAE returns tensor directly
            latents = latents
    
    # Apply FLUX scaling if available
    if hasattr(vae, 'config'):
        if hasattr(vae.config, 'shift_factor'):
            latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    
    B, C, H, W = latents.shape
    
    # Pack latents for FLUX: 2x2 packing
    # From [B, C, H, W] to [B, H/2 * W/2, C * 4]
    latents = latents.view(B, C, H // 2, 2, W // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    tokens = latents.reshape(B, (H // 2) * (W // 2), C * 4)
    
    # Create position IDs
    h_packed = H // 2
    w_packed = W // 2
    ids = torch.zeros(h_packed * w_packed, 3, device=device, dtype=dtype)
    for i in range(h_packed):
        for j in range(w_packed):
            idx = i * w_packed + j
            ids[idx, 0] = 0  # batch index
            ids[idx, 1] = i  # height
            ids[idx, 2] = j  # width
    
    return tokens, ids


def decode_latents_to_image(vae, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Decode FLUX latents back to image.
    
    Args:
        latents: Packed latent tokens [B, H*W/4, C*4]
        height: Original image height
        width: Original image width
    
    Returns:
        image: Tensor [B, H, W, C] in range [0, 1]
    """
    B = latents.shape[0]
    
    # Unpack latents: from [B, H/2 * W/2, C * 4] to [B, C, H, W]
    h_packed = height // 16  # VAE downscales 8x, then 2x packing
    w_packed = width // 16
    C = latents.shape[-1] // 4
    
    latents = latents.view(B, h_packed, w_packed, C, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(B, C, h_packed * 2, w_packed * 2)
    
    # Apply inverse FLUX scaling if available
    if hasattr(vae, 'config'):
        if hasattr(vae.config, 'scaling_factor'):
            latents = latents / vae.config.scaling_factor
        if hasattr(vae.config, 'shift_factor'):
            latents = latents + vae.config.shift_factor
    
    # Decode with VAE
    with torch.no_grad():
        image = vae.decode(latents)
        if hasattr(image, 'sample'):
            image = image.sample
    
    # Clamp and convert to BHWC
    image = image.clamp(0, 1)
    if image.shape[1] == 3:  # BCHW -> BHWC
        image = image.permute(0, 2, 3, 1)
    
    return image


def create_gaussian_weights(tile_size: int, device, dtype) -> torch.Tensor:
    """Create Gaussian weights for tile blending."""
    temp_d = torch.linspace(-2, 2, tile_size, dtype=dtype, device=device)
    gaussian_2d = torch.exp(-(temp_d[:, None] ** 2 + temp_d[None] ** 2))
    return gaussian_2d.flatten()


def compute_tile_coordinates(h: int, w: int, tile_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute tile coordinates for tiled processing.
    Uses 12.5% overlap between tiles for smooth blending.
    """
    overlap = tile_size // 8
    stride = tile_size - overlap
    
    h_coords = np.arange(0, max(1, h - tile_size + 1), stride, dtype=np.int32)
    w_coords = np.arange(0, max(1, w - tile_size + 1), stride, dtype=np.int32)
    
    # Ensure full coverage
    if len(h_coords) == 0:
        h_coords = np.array([0], dtype=np.int32)
    elif h_coords[-1] + tile_size < h:
        h_coords = np.append(h_coords, max(0, h - tile_size))
    
    if len(w_coords) == 0:
        w_coords = np.array([0], dtype=np.int32)
    elif w_coords[-1] + tile_size < w:
        w_coords = np.append(w_coords, max(0, w - tile_size))
    
    # Create grid
    h_grid, w_grid = np.meshgrid(h_coords, w_coords)
    return h_grid.ravel(), w_grid.ravel()


@contextmanager
def specify_lora_scale(model, adapter_name: str, active_adapters: List[str]):
    """
    Context manager to temporarily activate a specific LoRA adapter.
    
    For ComfyUI, we patch the model's LoRA weights to only use the specified adapter.
    """
    # This is a simplified version - full implementation would need to
    # integrate with ComfyUI's LoRA patching system
    try:
        yield
    finally:
        pass


def prepare_conditions(
    conditions: List[Condition],
    vae,
    device,
    dtype,
    pooled_embeds: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str], List[torch.Tensor]]:
    """
    Prepare condition latents for multi-branch generation.
    
    Returns:
        c_latents: List of encoded condition latents
        c_ids: List of position IDs for each condition
        c_adapters: List of adapter names
        c_projections: List of pooled projections
    """
    c_latents = []
    c_ids = []
    c_adapters = []
    c_projections = []
    
    for cond in conditions:
        tokens, ids = encode_image_to_latents(vae, cond.condition, device, dtype)
        
        # Apply position modifications
        if cond.position_delta:
            ids[:, 1] += cond.position_delta[0]
            ids[:, 2] += cond.position_delta[1]
        
        if cond.position_scale != 1.0:
            scale_bias = (cond.position_scale - 1.0) / 2
            ids[:, 1:] = ids[:, 1:] * cond.position_scale + scale_bias
        
        c_latents.append(tokens)
        c_ids.append(ids)
        c_adapters.append(cond.adapter)
        c_projections.append(pooled_embeds)
    
    return c_latents, c_ids, c_adapters, c_projections


def create_coord_to_idx_map(latent_ids: torch.Tensor) -> torch.Tensor:
    """Create a mapping from (h, w) coordinates to flat indices."""
    h_coords = latent_ids[:, 1].long()
    w_coords = latent_ids[:, 2].long()
    latent_h = h_coords.max().item() + 1
    latent_w = w_coords.max().item() + 1
    
    coord_to_idx = torch.full((latent_h, latent_w), -1, dtype=torch.long, device=latent_ids.device)
    coord_to_idx[h_coords, w_coords] = torch.arange(len(latent_ids), device=latent_ids.device)
    
    return coord_to_idx
