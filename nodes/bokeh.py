"""
BokehNet application node.

Applies the BokehNet LoRA to generate realistic bokeh effects.
This is Stage 2 of the Genfocus pipeline.

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


class BokehNetApply:
    """
    Apply BokehNet LoRA to generate bokeh effects.
    
    Takes a sharp image and a defocus map to produce realistic
    depth-of-field blur (bokeh) effects.
    
    Uses the prompt: "an excellent photo with a large aperture"
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
                "defocus_map": ("DEFOCUS_MAP",),
            },
            "optional": {
                "prompt": ("STRING", {
                    "default": "an excellent photo with a large aperture",
                    "multiline": True,
                    "tooltip": "Text prompt to guide bokeh generation"
                }),
                "steps": ("INT", {
                    "default": 28,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Number of denoising steps"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance scale"
                }),
                "seed": ("INT", {
                    "default": 1234,
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
                "latents": ("LATENT", {
                    "tooltip": "Optional: reuse latents from previous run for consistency"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("bokeh_image", "latents")
    FUNCTION = "apply_bokeh"
    CATEGORY = "Refocus/Bokeh"
    DESCRIPTION = "Apply BokehNet to generate realistic depth-of-field effects"
    
    def apply_bokeh(self, model, clip, vae, genfocus_loras, image, defocus_map,
                    prompt="an excellent photo with a large aperture",
                    steps=28, guidance_scale=1.0, seed=1234,
                    lora_strength=1.0, latents=None):
        
        # Note: Genfocus LoRAs require custom transformer_forward not yet implemented
        # Using base model with appropriate prompts for now
        
        # Get image dimensions
        B, H, W, C = image.shape
        original_h, original_w = H, W
        
        # Ensure dimensions are multiples of 16
        new_h = ((H + 15) // 16) * 16
        new_w = ((W + 15) // 16) * 16
        
        if new_h != H or new_w != W:
            # Pad image
            padded = torch.zeros(B, new_h, new_w, C, dtype=image.dtype, device=image.device)
            padded[:, :H, :W, :] = image
            image = padded
        
        device = mm.get_torch_device()
        
        # Get defocus map tensor
        dmf = defocus_map["map"]
        blur_strength = defocus_map["blur_strength"]
        
        # If blur strength is 0, just return the input image
        if blur_strength == 0:
            print("[BokehNet] Blur strength K=0, returning original image")
            image_bhwc = image.to(device)
            latent = vae.encode(image_bhwc[:, :, :, :3])
            if new_h != original_h or new_w != original_w:
                image = image[:, :original_h, :original_w, :]
            return (image, {"samples": latent})
        
        print(f"[BokehNet] Processing image: {new_w}x{new_h}")
        print(f"[BokehNet] Prompt: {prompt}")
        print(f"[BokehNet] Steps: {steps}, Guidance: {guidance_scale}, Seed: {seed}")
        print(f"[BokehNet] Blur strength K={blur_strength}")
        
        # Resize defocus map if needed
        if dmf.shape[1] != new_h or dmf.shape[2] != new_w:
            dmf = F.interpolate(
                dmf.unsqueeze(1).float(),  # B,H,W -> B,1,H,W
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # Back to B,H,W
        
        # Use base model (custom Genfocus LoRA application not yet implemented)
        model_with_lora = model
        if genfocus_loras.get("bokeh_path"):
            print(f"[BokehNet] Note: Using base model (custom LoRA application not yet implemented)")
        
        # Encode prompt
        tokens = clip.tokenize(prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        positive = [[cond, {"pooled_output": pooled}]]
        
        neg_prompt = "sharp everywhere, deep focus, small aperture, everything in focus"
        neg_tokens = clip.tokenize(neg_prompt)
        neg_cond, neg_pooled = clip.encode_from_tokens(neg_tokens, return_pooled=True)
        negative = [[neg_cond, {"pooled_output": neg_pooled}]]
        
        # Encode input image to latent
        image_bhwc = image.to(device)
        input_latent = vae.encode(image_bhwc[:, :, :, :3])
        
        # Prepare noise
        noise = comfy.sample.prepare_noise(input_latent, seed)
        
        # Calculate denoise strength based on defocus
        # More defocus = more denoise, less = preserve original
        avg_defocus = dmf.mean().item()
        # Scale denoise: min 0.3, max 0.8 based on average defocus
        denoise = 0.3 + (0.5 * min(1.0, avg_defocus))
        print(f"[BokehNet] Average defocus: {avg_defocus:.3f}, Denoise: {denoise:.3f}")
        
        # Set up sampler
        sampler = comfy.samplers.KSampler(
            model_with_lora,
            steps=steps,
            device=device,
            sampler=comfy.samplers.sampler_object("euler"),
            scheduler="normal",
            denoise=denoise,
            model_options={}
        )
        
        # Sample
        print(f"[BokehNet] Starting denoising...")
        samples = sampler.sample(
            noise,
            positive,
            negative,
            cfg=guidance_scale,
            latent_image=input_latent,
            start_step=None,
            last_step=None,
            force_full_denoise=False,
            denoise_mask=None,
            callback=None,
            disable_pbar=False,
            seed=seed
        )
        
        # Blend output with input using defocus map as mask
        # Focus area (low defocus) keeps original, blur area (high defocus) gets generated
        dmf_latent = F.interpolate(
            dmf.unsqueeze(1).float().to(device),
            size=(samples.shape[2], samples.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        
        # Blend: samples where defocus high, input_latent where defocus low
        blended_samples = input_latent * (1 - dmf_latent) + samples * dmf_latent
        
        print(f"[BokehNet] Decoding output...")
        
        # Decode output
        output_image = vae.decode(blended_samples)
        
        # Crop back to original size if we padded
        if new_h != original_h or new_w != original_w:
            output_image = output_image[:, :original_h, :original_w, :]
        
        latent_dict = {"samples": blended_samples}
        
        print(f"[BokehNet] Complete!")
        
        return (output_image, latent_dict)
