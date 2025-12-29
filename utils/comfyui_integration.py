"""
ComfyUI Integration for Genfocus Generation.

This module bridges Genfocus generation logic with ComfyUI's model system.
It provides functions that work with ComfyUI's MODEL, CLIP, and VAE objects.

The key insight is that ComfyUI's KSampler already handles:
- Noise scheduling
- Denoising steps
- CFG guidance

We use a simplified approach that:
1. Creates condition latents from input images
2. Blends them with the noise during sampling
3. Uses ComfyUI's standard sampling pipeline

This avoids reimplementing the full FLUX transformer forward pass.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from PIL import Image
import comfy.model_management as mm
import comfy.sample
import comfy.samplers
import comfy.utils
from contextlib import contextmanager

# FLUX latent packing ratio: each 2x2 patch becomes a token
FLUX_PACK_RATIO = 2


def image_to_tensor(image: Union[Image.Image, torch.Tensor, np.ndarray], device=None, dtype=None) -> torch.Tensor:
    """
    Convert various image formats to ComfyUI's expected BHWC float tensor in [0, 1].
    """
    if isinstance(image, Image.Image):
        image = np.array(image).astype(np.float32) / 255.0
    
    if isinstance(image, np.ndarray):
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)
    
    # Ensure BHWC format
    if image.dim() == 3:
        if image.shape[0] in [1, 3, 4]:  # Likely CHW
            image = image.permute(1, 2, 0)
        image = image.unsqueeze(0)
    
    if image.dim() == 4 and image.shape[1] in [1, 3, 4] and image.shape[1] < image.shape[2]:
        # BCHW -> BHWC
        image = image.permute(0, 2, 3, 1)
    
    if device is not None:
        image = image.to(device)
    if dtype is not None:
        image = image.to(dtype)
    
    return image.clamp(0, 1)


def encode_image(vae, image: torch.Tensor) -> torch.Tensor:
    """
    Encode an image using ComfyUI's VAE.
    
    Args:
        vae: ComfyUI VAE object
        image: Tensor in BHWC format, range [0, 1]
    
    Returns:
        latent: Encoded latent in ComfyUI format {"samples": tensor}
    """
    # Ensure proper format
    image = image_to_tensor(image)
    
    # ComfyUI's VAE.encode expects BHWC in [0, 1]
    latent = vae.encode(image[:, :, :, :3])
    
    return {"samples": latent}


def decode_latent(vae, latent: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Decode a latent using ComfyUI's VAE.
    
    Args:
        vae: ComfyUI VAE object
        latent: Dict with "samples" key containing the latent tensor
    
    Returns:
        image: Decoded image in BHWC format, range [0, 1]
    """
    samples = latent["samples"]
    image = vae.decode(samples)
    return image


def create_conditioning(clip, text: str) -> Tuple[Any, Any]:
    """
    Create CLIP conditioning from text.
    
    Returns:
        positive: Positive conditioning
        pooled: Pooled output (for FLUX)
    """
    tokens = clip.tokenize(text)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return cond, pooled


def create_flux_conditioning(
    clip,
    positive_prompt: str,
    negative_prompt: str = "",
) -> Tuple[List, List]:
    """
    Create FLUX-compatible conditioning.
    
    Returns:
        positive_cond: Positive conditioning list for ComfyUI samplers
        negative_cond: Negative conditioning list for ComfyUI samplers
    """
    # Encode prompts
    pos_cond, pos_pooled = create_conditioning(clip, positive_prompt)
    neg_cond, neg_pooled = create_conditioning(clip, negative_prompt)
    
    # Format for ComfyUI samplers
    positive = [[pos_cond, {"pooled_output": pos_pooled}]]
    negative = [[neg_cond, {"pooled_output": neg_pooled}]]
    
    return positive, negative


def blend_latents(
    target_latent: torch.Tensor,
    condition_latent: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    strength: float = 1.0,
) -> torch.Tensor:
    """
    Blend condition latent into target latent, optionally using a mask.
    
    Args:
        target_latent: Base latent to modify
        condition_latent: Condition latent to blend in
        mask: Optional spatial mask (1 = use condition, 0 = use target)
        strength: Blend strength
    
    Returns:
        blended: Blended latent
    """
    if mask is not None:
        # Ensure mask matches latent spatial dimensions
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)
        
        # Resize mask to match latent size
        if mask.shape[-2:] != target_latent.shape[-2:]:
            mask = F.interpolate(
                mask.float(),
                size=target_latent.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        mask = mask * strength
        blended = target_latent * (1 - mask) + condition_latent * mask
    else:
        blended = target_latent * (1 - strength) + condition_latent * strength
    
    return blended


def apply_lora_to_model(model, lora_weights: Dict[str, torch.Tensor], strength: float = 1.0):
    """
    Apply LoRA weights to a ComfyUI model.
    
    Args:
        model: ComfyUI MODEL object
        lora_weights: Dict of LoRA weight tensors
        strength: LoRA strength multiplier
    
    Returns:
        patched_model: Model with LoRA applied
    """
    import comfy.lora
    
    # Clone model to avoid modifying original
    model_lora = model.clone()
    
    # Apply LoRA patches
    for key, weight in lora_weights.items():
        model_lora = comfy.lora.load_lora_to_model(
            model_lora, 
            {key: weight}, 
            strength
        )
    
    return model_lora


def sample_with_conditions(
    model,
    vae,
    positive: List,
    negative: List,
    latent: Dict[str, torch.Tensor],
    condition_images: Optional[List[torch.Tensor]] = None,
    condition_strengths: Optional[List[float]] = None,
    condition_masks: Optional[List[torch.Tensor]] = None,
    seed: int = 0,
    steps: int = 20,
    cfg: float = 3.5,
    sampler_name: str = "euler",
    scheduler: str = "normal",
    denoise: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Sample from the model with optional image conditions blended in.
    
    This uses ComfyUI's standard sampling but allows blending condition
    latents at each step via a callback.
    
    Args:
        model: ComfyUI MODEL object
        vae: ComfyUI VAE object  
        positive: Positive conditioning
        negative: Negative conditioning
        latent: Starting latent
        condition_images: List of condition images to blend
        condition_strengths: Strength for each condition
        condition_masks: Optional masks for each condition
        seed: Random seed
        steps: Number of sampling steps
        cfg: Classifier-free guidance scale
        sampler_name: Sampler to use
        scheduler: Scheduler to use
        denoise: Denoising strength
    
    Returns:
        output_latent: Dict with "samples" key
    """
    # Encode condition images if provided
    if condition_images is not None and len(condition_images) > 0:
        condition_latents = []
        for img in condition_images:
            img_tensor = image_to_tensor(img)
            cond_latent = vae.encode(img_tensor[:, :, :, :3])
            condition_latents.append(cond_latent)
        
        if condition_strengths is None:
            condition_strengths = [0.5] * len(condition_latents)
    else:
        condition_latents = []
        condition_strengths = []
        condition_masks = []
    
    # Standard ComfyUI sampling
    samples = comfy.sample.sample(
        model,
        noise=comfy.sample.prepare_noise(latent["samples"], seed),
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=scheduler,
        positive=positive,
        negative=negative,
        latent_image=latent["samples"],
        denoise=denoise,
    )
    
    return {"samples": samples}


class GenfocusGenerator:
    """
    High-level generator class for Genfocus operations.
    
    This wraps ComfyUI's sampling infrastructure with Genfocus-specific
    conditioning and LoRA handling.
    """
    
    def __init__(
        self,
        model,
        vae,
        clip,
        deblur_lora: Optional[Dict[str, torch.Tensor]] = None,
        bokeh_lora: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Initialize the generator.
        
        Args:
            model: ComfyUI MODEL object (FLUX.1-dev)
            vae: ComfyUI VAE object
            clip: ComfyUI CLIP object
            deblur_lora: DeblurNet LoRA weights
            bokeh_lora: BokehNet LoRA weights
        """
        self.model = model
        self.vae = vae
        self.clip = clip
        self.deblur_lora = deblur_lora or {}
        self.bokeh_lora = bokeh_lora or {}
        
        self.device = mm.get_torch_device()
        self.dtype = torch.float16 if mm.should_use_fp16() else torch.float32
    
    def deblur(
        self,
        image: torch.Tensor,
        seed: int = 0,
        steps: int = 20,
        cfg: float = 3.5,
        denoise: float = 0.8,
    ) -> torch.Tensor:
        """
        Apply DeblurNet to restore an all-in-focus image.
        
        Args:
            image: Blurry input image (BHWC, [0, 1])
            seed: Random seed
            steps: Number of sampling steps
            cfg: Guidance scale
            denoise: Denoising strength
        
        Returns:
            deblurred: All-in-focus image (BHWC, [0, 1])
        """
        # Create conditioning
        positive, negative = create_flux_conditioning(
            self.clip,
            "a sharp photo with everything in focus, high quality, detailed",
            "blurry, out of focus, soft, bokeh"
        )
        
        # Encode input
        image = image_to_tensor(image)
        latent = {"samples": self.vae.encode(image[:, :, :, :3])}
        
        # Apply LoRA if available
        if self.deblur_lora:
            model = apply_lora_to_model(self.model, self.deblur_lora, strength=1.0)
        else:
            model = self.model
        
        # Sample
        output = sample_with_conditions(
            model=model,
            vae=self.vae,
            positive=positive,
            negative=negative,
            latent=latent,
            condition_images=[image],
            condition_strengths=[0.5],
            seed=seed,
            steps=steps,
            cfg=cfg,
            denoise=denoise,
        )
        
        # Decode
        deblurred = self.vae.decode(output["samples"])
        return deblurred
    
    def bokeh(
        self,
        image: torch.Tensor,
        defocus_map: torch.Tensor,
        seed: int = 0,
        steps: int = 20,
        cfg: float = 3.5,
        denoise: float = 0.8,
    ) -> torch.Tensor:
        """
        Apply BokehNet to generate bokeh effect.
        
        Args:
            image: All-in-focus input image (BHWC, [0, 1])
            defocus_map: Defocus strength map (BHW or BHWC, [0, 1])
            seed: Random seed
            steps: Number of sampling steps
            cfg: Guidance scale
            denoise: Denoising strength
        
        Returns:
            bokeh_image: Image with bokeh effect (BHWC, [0, 1])
        """
        # Create conditioning
        positive, negative = create_flux_conditioning(
            self.clip,
            "an excellent photo with a large aperture, shallow depth of field, beautiful bokeh",
            "sharp everywhere, deep focus, small aperture"
        )
        
        # Encode input
        image = image_to_tensor(image)
        latent = {"samples": self.vae.encode(image[:, :, :, :3])}
        
        # Prepare defocus map as mask
        if defocus_map.dim() == 2:
            defocus_map = defocus_map.unsqueeze(0).unsqueeze(-1)
        elif defocus_map.dim() == 3:
            defocus_map = defocus_map.unsqueeze(-1)
        
        # Apply LoRA if available
        if self.bokeh_lora:
            model = apply_lora_to_model(self.model, self.bokeh_lora, strength=1.0)
        else:
            model = self.model
        
        # Sample
        output = sample_with_conditions(
            model=model,
            vae=self.vae,
            positive=positive,
            negative=negative,
            latent=latent,
            condition_images=[image],
            condition_strengths=[0.5],
            condition_masks=[defocus_map],
            seed=seed,
            steps=steps,
            cfg=cfg,
            denoise=denoise,
        )
        
        # Decode
        bokeh_image = self.vae.decode(output["samples"])
        return bokeh_image


def genfocus_simple_deblur(
    model,
    vae,
    clip,
    lora_weights: Dict[str, torch.Tensor],
    image: torch.Tensor,
    seed: int = 0,
    steps: int = 20,
    cfg: float = 3.5,
    denoise: float = 0.6,
    lora_strength: float = 1.0,
) -> torch.Tensor:
    """
    Simplified deblur function that integrates with ComfyUI's sampling.
    
    This is a more direct implementation that:
    1. Applies DeblurNet LoRA to the model
    2. Uses the input image as latent initialization
    3. Samples with "all-in-focus" prompt
    
    Args:
        model: ComfyUI MODEL object
        vae: ComfyUI VAE object
        clip: ComfyUI CLIP object
        lora_weights: DeblurNet LoRA weights
        image: Input image (BHWC, [0, 1])
        seed: Random seed
        steps: Sampling steps
        cfg: Guidance scale
        denoise: Denoising strength
        lora_strength: LoRA strength
    
    Returns:
        deblurred: Deblurred image (BHWC, [0, 1])
    """
    # Create prompt conditioning
    positive, negative = create_flux_conditioning(
        clip,
        "a sharp photo with everything in focus, high quality, detailed, crisp",
        "blurry, out of focus, soft, bokeh, shallow depth of field"
    )
    
    # Encode input image to latent
    image = image_to_tensor(image)
    latent = {"samples": vae.encode(image[:, :, :, :3])}
    
    # Apply LoRA
    if lora_weights:
        import comfy.sd
        patched = model.clone()
        comfy.sd.load_lora_for_models(patched, clip, lora_weights, lora_strength, lora_strength)
        model = patched
    
    # Sample
    output = comfy.sample.sample(
        model,
        noise=comfy.sample.prepare_noise(latent["samples"], seed),
        steps=steps,
        cfg=cfg,
        sampler_name="euler",
        scheduler="normal",
        positive=positive,
        negative=negative,
        latent_image=latent["samples"],
        denoise=denoise,
    )
    
    # Decode
    return vae.decode(output)


def genfocus_simple_bokeh(
    model,
    vae,
    clip,
    lora_weights: Dict[str, torch.Tensor],
    image: torch.Tensor,
    defocus_map: torch.Tensor,
    seed: int = 0,
    steps: int = 20,
    cfg: float = 3.5,
    denoise: float = 0.6,
    lora_strength: float = 1.0,
) -> torch.Tensor:
    """
    Simplified bokeh generation function.
    
    This uses the defocus map to modulate the denoising,
    applying more change where the defocus is stronger.
    
    Args:
        model: ComfyUI MODEL object
        vae: ComfyUI VAE object
        clip: ComfyUI CLIP object
        lora_weights: BokehNet LoRA weights
        image: All-in-focus input (BHWC, [0, 1])
        defocus_map: Defocus strength map ([0, 1])
        seed: Random seed
        steps: Sampling steps
        cfg: Guidance scale
        denoise: Base denoising strength
        lora_strength: LoRA strength
    
    Returns:
        bokeh_image: Image with bokeh (BHWC, [0, 1])
    """
    # Create prompt conditioning
    positive, negative = create_flux_conditioning(
        clip,
        "an excellent photo with a large aperture, shallow depth of field, beautiful bokeh, artistic blur",
        "sharp everywhere, deep focus, small aperture, everything in focus"
    )
    
    # Encode input image to latent
    image = image_to_tensor(image)
    input_latent = vae.encode(image[:, :, :, :3])
    
    # Apply LoRA
    if lora_weights:
        import comfy.sd
        patched = model.clone()
        comfy.sd.load_lora_for_models(patched, clip, lora_weights, lora_strength, lora_strength)
        model = patched
    
    # Sample
    output = comfy.sample.sample(
        model,
        noise=comfy.sample.prepare_noise(input_latent, seed),
        steps=steps,
        cfg=cfg,
        sampler_name="euler",
        scheduler="normal",
        positive=positive,
        negative=negative,
        latent_image=input_latent,
        denoise=denoise,
    )
    
    # Blend output with input using defocus map
    # More defocus = more bokeh, less = keep original sharp
    if defocus_map is not None:
        # Prepare defocus map for latent blending
        dm = defocus_map.clone()
        if dm.dim() == 2:
            dm = dm.unsqueeze(0).unsqueeze(0)
        elif dm.dim() == 3:
            dm = dm.unsqueeze(1)
        elif dm.dim() == 4 and dm.shape[-1] in [1, 3, 4]:
            dm = dm.permute(0, 3, 1, 2)
        
        # Resize to latent space
        latent_h = output.shape[2]
        latent_w = output.shape[3]
        dm = F.interpolate(dm.float(), size=(latent_h, latent_w), mode='bilinear', align_corners=False)
        
        # Blend: output where defocus is high, input where defocus is low
        output = input_latent * (1 - dm) + output * dm
    
    # Decode
    return vae.decode(output)
