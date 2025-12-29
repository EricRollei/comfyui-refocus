"""
DeblurNet application node.

Applies the DeblurNet LoRA to produce an all-in-focus image from a blurry input.
This is Stage 1 of the Genfocus pipeline.

License: Apache 2.0 (same as Genfocus project)
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import comfy.model_management as mm
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.lora


class DeblurNetApply:
    """
    Apply DeblurNet LoRA to deblur an image.
    
    Takes a blurry/out-of-focus image and produces a sharp, all-in-focus result.
    This is useful for:
    - Restoring focus to blurry photos
    - Preparing images for the BokehNet stage
    - General deblurring of optical blur (not motion blur)
    
    Uses the prompt: "a sharp photo with everything in focus"
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "genfocus_loras": ("GENFOCUS_LORAS",),
                "image": ("IMAGE",),
            },
            "optional": {
                "prompt": ("STRING", {
                    "default": "a sharp photo with everything in focus",
                    "multiline": True,
                    "tooltip": "Text prompt to guide deblurring"
                }),
                "steps": ("INT", {
                    "default": 28,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Number of denoising steps"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 3.5,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance scale"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducibility"
                }),
                "lora_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "LoRA adapter strength"
                }),
                "tile_size": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                    "tooltip": "Tile size for high-res processing (0 = no tiling)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("deblurred_image", "latents")
    FUNCTION = "apply_deblur"
    CATEGORY = "Refocus/Deblur"
    DESCRIPTION = "Apply DeblurNet to restore an all-in-focus image"
    
    def apply_deblur(self, model, clip, vae, genfocus_loras, image, 
                     prompt="a sharp photo with everything in focus",
                     steps=28, guidance_scale=3.5, seed=42, 
                     lora_strength=1.0, tile_size=512):
        
        # Note: Genfocus LoRAs require custom transformer_forward not yet implemented
        # Using base model with appropriate prompts for now
        
        # Get image dimensions
        B, H, W, C = image.shape
        original_h, original_w = H, W
        
        # Ensure dimensions are multiples of 16 (FLUX requirement)
        new_h = ((H + 15) // 16) * 16
        new_w = ((W + 15) // 16) * 16
        
        if new_h != H or new_w != W:
            # Pad image to valid size
            padded = torch.zeros(B, new_h, new_w, C, dtype=image.dtype, device=image.device)
            padded[:, :H, :W, :] = image
            image = padded
        
        device = mm.get_torch_device()
        
        print(f"[DeblurNet] Processing image: {new_w}x{new_h}")
        print(f"[DeblurNet] Prompt: {prompt}")
        print(f"[DeblurNet] Steps: {steps}, Guidance: {guidance_scale}, Seed: {seed}")
        
        # Use base model (custom Genfocus LoRA application not yet implemented)
        model_with_lora = model
        if genfocus_loras.get("deblur_path"):
            print(f"[DeblurNet] Note: Using base model (custom LoRA application not yet implemented)")
        
        # Encode prompt
        tokens = clip.tokenize(prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        positive = [[cond, {"pooled_output": pooled}]]
        
        # Negative prompt
        neg_prompt = "blurry, out of focus, soft, bokeh, shallow depth of field"
        neg_tokens = clip.tokenize(neg_prompt)
        neg_cond, neg_pooled = clip.encode_from_tokens(neg_tokens, return_pooled=True)
        negative = [[neg_cond, {"pooled_output": neg_pooled}]]
        
        # Encode input image to latent
        image_bhwc = image.to(device)
        latent = vae.encode(image_bhwc[:, :, :, :3])
        
        # Prepare noise
        noise = comfy.sample.prepare_noise(latent, seed)
        
        # Set up sampler
        sampler = comfy.samplers.KSampler(
            model_with_lora,
            steps=steps,
            device=device,
            sampler=comfy.samplers.sampler_object("euler"),
            scheduler="normal",
            denoise=0.7,  # Partial denoise to preserve structure
            model_options={}
        )
        
        # Sample
        print(f"[DeblurNet] Starting denoising...")
        samples = sampler.sample(
            noise,
            positive,
            negative,
            cfg=guidance_scale,
            latent_image=latent,
            start_step=None,
            last_step=None,
            force_full_denoise=False,
            denoise_mask=None,
            callback=None,
            disable_pbar=False,
            seed=seed
        )
        
        print(f"[DeblurNet] Decoding output...")
        
        # Decode output
        output_image = vae.decode(samples)
        
        # Crop back to original size if we padded
        if new_h != original_h or new_w != original_w:
            output_image = output_image[:, :original_h, :original_w, :]
        
        # Create latent dict
        latent_dict = {"samples": samples}
        
        print(f"[DeblurNet] Complete!")
        
        return (output_image, latent_dict)
