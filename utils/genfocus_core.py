"""
Core Genfocus generation utilities.

This module contains the ported Genfocus pipeline logic including:
- Multi-conditional generation with LoRA switching
- Custom transformer forward pass
- Tiled denoising for high-resolution images

Based on: https://github.com/rayray9999/Genfocus/blob/main/Genfocus/pipeline/flux.py
License: Apache 2.0
"""

import torch
import numpy as np
from typing import List, Union, Optional, Dict, Any, Tuple
from contextlib import contextmanager
from PIL import Image


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
    - position_delta: Offset for position embeddings
    - position_scale: Scale factor for position embeddings
    - latent_mask: Optional mask for partial conditioning
    """
    
    def __init__(
        self,
        condition: Union[Image.Image, torch.Tensor],
        adapter_setting: Union[str, dict],
        position_delta: Optional[List[int]] = None,
        position_scale: float = 1.0,
        latent_mask: Optional[torch.Tensor] = None,
        is_complement: bool = False,
        no_preprocess: bool = False,
    ) -> None:
        self.condition = condition
        self.adapter = adapter_setting
        self.position_delta = position_delta
        self.position_scale = position_scale
        self.latent_mask = (
            latent_mask.T.reshape(-1) if latent_mask is not None else None
        )
        self.is_complement = is_complement
        self.no_preprocess = no_preprocess
    
    def encode(self, vae, image_processor, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode the condition image to latent space."""
        if isinstance(self.condition, Image.Image):
            # Preprocess PIL image
            if not self.no_preprocess:
                images = image_processor.preprocess(self.condition)
            else:
                images = torch.from_numpy(np.array(self.condition)).float() / 255.0
                images = images.permute(2, 0, 1).unsqueeze(0)
        else:
            images = self.condition
        
        images = images.to(device).to(dtype)
        
        # Encode with VAE
        latents = vae.encode(images).latent_dist.sample()
        
        # Apply FLUX-specific scaling
        if hasattr(vae.config, 'shift_factor'):
            latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
        
        # Create position IDs
        B, C, H, W = latents.shape
        ids = self._create_position_ids(H, W, device, dtype)
        
        # Apply position modifications
        if self.position_delta is not None:
            ids[:, 1] += self.position_delta[0]
            ids[:, 2] += self.position_delta[1]
        
        if self.position_scale != 1.0:
            scale_bias = (self.position_scale - 1.0) / 2
            ids[:, 1:] *= self.position_scale
            ids[:, 1:] += scale_bias
        
        # Pack latents for FLUX format
        tokens = self._pack_latents(latents)
        
        if self.latent_mask is not None:
            tokens = tokens[:, self.latent_mask]
            ids = ids[self.latent_mask]
        
        return tokens, ids
    
    def _create_position_ids(self, h: int, w: int, device, dtype) -> torch.Tensor:
        """Create position IDs for FLUX."""
        ids = torch.zeros(h * w, 3, device=device, dtype=dtype)
        for i in range(h):
            for j in range(w):
                ids[i * w + j, 1] = i
                ids[i * w + j, 2] = j
        return ids
    
    def _pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Pack latents from (B,C,H,W) to (B,H*W,C) for FLUX."""
        B, C, H, W = latents.shape
        # FLUX uses 2x2 packing
        latents = latents.view(B, C, H // 2, 2, W // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(B, (H // 2) * (W // 2), C * 4)
        return latents


@contextmanager
def specify_lora(lora_modules: List, specified_lora: str):
    """
    Context manager for temporarily activating a specific LoRA adapter.
    
    This allows dynamic switching between deblurNet and bokehNet
    during the multi-branch forward pass.
    """
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except ImportError:
        # PEFT not installed - just yield
        yield
        return
    
    # Filter valid LoRA modules
    valid_lora_modules = [m for m in lora_modules if isinstance(m, BaseTunerLayer)]
    
    if not valid_lora_modules:
        yield
        return
    
    # Save original scales
    original_scales = [
        {
            adapter: module.scaling[adapter]
            for adapter in module.active_adapters
            if adapter in module.scaling
        }
        for module in valid_lora_modules
    ]
    
    # Set scales: 1 for specified adapter, 0 for others
    for module in valid_lora_modules:
        for adapter in module.active_adapters:
            if adapter in module.scaling:
                module.scaling[adapter] = 1 if adapter == specified_lora else 0
    
    try:
        yield
    finally:
        # Restore original scales
        for module, scales in zip(valid_lora_modules, original_scales):
            for adapter in module.active_adapters:
                if adapter in module.scaling:
                    module.scaling[adapter] = scales[adapter]


def create_gaussian_weights(tile_size: int, device, dtype) -> torch.Tensor:
    """Create Gaussian weights for tile blending."""
    temp_d = torch.linspace(-2, 2, tile_size, dtype=dtype, device=device)
    gaussian_2d = torch.exp(-(temp_d[:, None] ** 2 + temp_d[None] ** 2))
    return gaussian_2d.flatten()


def compute_tile_coordinates(h: int, w: int, tile_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute tile coordinates for tiled processing.
    
    Uses 12.5% overlap between tiles for smooth blending.
    """
    overlap = tile_size // 8
    stride = tile_size - overlap
    
    h_coords = np.arange(0, max(1, h - tile_size + 1), stride, dtype=np.int32)
    w_coords = np.arange(0, max(1, w - tile_size + 1), stride, dtype=np.int32)
    
    # Ensure full coverage
    if len(h_coords) == 0 or h_coords[-1] + tile_size < h:
        h_coords = np.append(h_coords, max(0, h - tile_size))
    if len(w_coords) == 0 or w_coords[-1] + tile_size < w:
        w_coords = np.append(w_coords, max(0, w - tile_size))
    
    # Create grid
    h_grid, w_grid = np.meshgrid(h_coords, w_coords)
    return h_grid.ravel(), w_grid.ravel()


def create_group_mask(num_conditions: int) -> torch.Tensor:
    """
    Create attention group mask for multi-branch processing.
    
    This mask prevents cross-attention between different condition branches
    while allowing attention between the main generation and text branches.
    """
    branch_n = num_conditions + 2  # +2 for text and image branches
    group_mask = torch.ones([branch_n, branch_n], dtype=torch.bool)
    
    # Disable attention between different condition branches
    if num_conditions > 0:
        group_mask[2:, 2:] = torch.diag(torch.tensor([1] * num_conditions))
    
    return group_mask
