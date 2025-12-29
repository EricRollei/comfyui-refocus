"""
Condition class for Genfocus multi-conditional generation.
Ported from rayray9999/Genfocus
"""

import torch
import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional

from diffusers.pipelines import FluxPipeline


def encode_images(pipeline: FluxPipeline, images: Union[Image.Image, torch.Tensor], no_preprocess: bool = False):
    """
    Encodes the images into tokens and ids for FLUX pipeline.
    
    Args:
        pipeline: FluxPipeline instance
        images: PIL Image or torch tensor to encode
        no_preprocess: If True, skip image preprocessing (for pre-processed tensors like defocus maps)
    
    Returns:
        Tuple of (image_tokens, image_ids)
    """
    if not no_preprocess:
        images = pipeline.image_processor.preprocess(images)
    
    images = images.to(pipeline.device).to(pipeline.dtype)
    images = pipeline.vae.encode(images).latent_dist.sample()
    images = (
        images - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor
    
    images_tokens = pipeline._pack_latents(images, *images.shape)
    images_ids = pipeline._prepare_latent_image_ids(
        images.shape[0],
        images.shape[2],
        images.shape[3],
        pipeline.device,
        pipeline.dtype,
    )
    
    if images_tokens.shape[1] != images_ids.shape[0]:
        images_ids = pipeline._prepare_latent_image_ids(
            images.shape[0],
            images.shape[2] // 2,
            images.shape[3] // 2,
            pipeline.device,
            pipeline.dtype,
        )
    
    return images_tokens, images_ids


class Condition:
    """
    Represents a conditioning input for Genfocus generation.
    
    Each condition can be an image (PIL or tensor) with an associated adapter (LoRA).
    The position_delta and position_scale allow offsetting the condition's position embeddings.
    """
    
    def __init__(
        self,
        condition: Union[Image.Image, torch.Tensor],
        adapter_setting: Union[str, dict],
        position_delta: Optional[list] = None,
        position_scale: float = 1.0,
        latent_mask: Optional[torch.Tensor] = None,
        is_complement: bool = False,
        no_preprocess: bool = False,
    ) -> None:
        """
        Initialize a Condition.
        
        Args:
            condition: The conditioning image (PIL Image or torch.Tensor)
            adapter_setting: LoRA adapter name ("deblurring" or "bokeh") or dict of settings
            position_delta: [height_offset, width_offset] for position embeddings
            position_scale: Scale factor for position embeddings
            latent_mask: Optional mask for sparse conditioning
            is_complement: If True, this condition is a complement (for OminiControl2 style)
            no_preprocess: If True, skip VAE preprocessing (for tensor inputs like defocus maps)
        """
        self.condition = condition
        self.adapter = adapter_setting
        self.position_delta = position_delta
        self.position_scale = position_scale
        self.latent_mask = (
            latent_mask.T.reshape(-1) if latent_mask is not None else None
        )
        self.is_complement = is_complement
        self.no_preprocess = no_preprocess
    
    def encode(
        self, pipe: FluxPipeline, empty: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the condition into latent tokens and position IDs.
        
        Args:
            pipe: FluxPipeline instance
            empty: If True, encode an empty (black) image instead
        
        Returns:
            Tuple of (tokens, ids) - latent tokens and position IDs
        """
        # Create empty image if requested
        if isinstance(self.condition, Image.Image):
            condition_empty = Image.new("RGB", self.condition.size, (0, 0, 0))
        elif torch.is_tensor(self.condition):
            H, W = self.condition.shape[-2], self.condition.shape[-1]
            condition_empty = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8), "RGB")
        else:
            raise ValueError(f"Unsupported condition type: {type(self.condition)}")
        
        # Encode the condition (or empty version)
        tokens, ids = encode_images(
            pipe, 
            condition_empty if empty else self.condition,
            no_preprocess=self.no_preprocess
        )
        
        # Apply position offset
        if self.position_delta is not None:
            ids[:, 1] += self.position_delta[0]
            ids[:, 2] += self.position_delta[1]
        
        # Apply position scaling
        if self.position_scale != 1.0:
            scale_bias = (self.position_scale - 1.0) / 2
            ids[:, 1:] *= self.position_scale
            ids[:, 1:] += scale_bias
        
        # Apply latent mask if present
        if self.latent_mask is not None:
            tokens = tokens[:, self.latent_mask]
            ids = ids[self.latent_mask]
        
        return tokens, ids
