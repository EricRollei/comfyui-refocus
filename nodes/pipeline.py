"""
Genfocus full pipeline node.

Provides a single-node interface for the complete Genfocus workflow:
1. Deblur input image (Stage 1 - DeblurNet)
2. Estimate depth (DepthPro)
3. Compute defocus map from focus point
4. Generate bokeh effects (Stage 2 - BokehNet)

License: Apache 2.0 (same as Genfocus project)
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import comfy.model_management as mm
import comfy.sample
import comfy.samplers
import comfy.lora


class GenfocusPipeline:
    """
    Complete Genfocus pipeline in a single node.
    
    This convenience node runs the full two-stage pipeline:
    - Stage 1: Deblur the input to get an all-in-focus image
    - Stage 2: Apply depth-based bokeh effects
    
    For more control, use the individual nodes:
    - DeblurNetApply
    - DepthProEstimate
    - ComputeDefocusMap
    - BokehNetApply
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "genfocus_loras": ("GENFOCUS_LORAS",),
                "depth_model": ("DEPTH_PRO_MODEL",),
                "image": ("IMAGE",),
            },
            "optional": {
                "blur_strength": ("FLOAT", {
                    "default": 20.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.5,
                    "tooltip": "K value - bokeh intensity (0 = deblur only)"
                }),
                "focus_x_percent": ("FLOAT", {
                    "default": 50.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "X coordinate of focus point (% of width)"
                }),
                "focus_y_percent": ("FLOAT", {
                    "default": 50.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Y coordinate of focus point (% of height)"
                }),
                "deblur_steps": ("INT", {
                    "default": 28,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Denoising steps for deblurring stage"
                }),
                "bokeh_steps": ("INT", {
                    "default": 28,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Denoising steps for bokeh stage"
                }),
                "deblur_denoise": ("FLOAT", {
                    "default": 0.35,
                    "min": 0.05,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Denoise strength for deblur. Lower = preserve original, Higher = more regeneration"
                }),
                "bokeh_denoise_max": ("FLOAT", {
                    "default": 0.35,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Max denoise for bokeh stage. Lower = preserve identity better"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducibility"
                }),
                "skip_deblur": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Skip deblurring if input is already sharp"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("final_output", "deblurred", "depth_map", "defocus_map")
    FUNCTION = "run_pipeline"
    CATEGORY = "Refocus/Pipeline"
    DESCRIPTION = "Run the complete Genfocus pipeline (deblur + bokeh)"
    
    def run_pipeline(self, model, clip, vae, genfocus_loras, depth_model,
                     image, blur_strength=20.0, focus_x_percent=50.0, 
                     focus_y_percent=50.0, deblur_steps=28, bokeh_steps=28,
                     deblur_denoise=0.35, bokeh_denoise_max=0.35,
                     seed=42, skip_deblur=False):
        
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
            H, W = new_h, new_w
        
        device = mm.get_torch_device()
        
        print(f"[Genfocus Pipeline] Input: {W}x{H}")
        print(f"[Genfocus Pipeline] Blur strength K={blur_strength}")
        print(f"[Genfocus Pipeline] Focus point: ({focus_x_percent}%, {focus_y_percent}%)")
        
        # Get LoRA paths (not raw weights - we'll use ComfyUI's loader)
        deblur_lora_path = genfocus_loras.get("deblur_path")
        bokeh_lora_path = genfocus_loras.get("bokeh_path")
        
        # =========== STAGE 1: Deblur ===========
        if skip_deblur:
            print("[Genfocus Pipeline] Skipping deblur stage")
            deblurred = image
        else:
            print("[Genfocus Pipeline] Stage 1: Deblurring...")
            
            # Use base model - Genfocus LoRAs require custom transformer_forward
            # which isn't compatible with standard ComfyUI LoRA application
            model_deblur = model
            if deblur_lora_path:
                print(f"[Genfocus Pipeline] Note: Using base model (custom LoRA application not yet implemented)")
            
            # Create deblur conditioning
            deblur_prompt = "a sharp photo with everything in focus, high quality, detailed"
            tokens = clip.tokenize(deblur_prompt)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            positive = [[cond, {"pooled_output": pooled}]]
            
            neg_tokens = clip.tokenize("blurry, out of focus, soft, bokeh")
            neg_cond, neg_pooled = clip.encode_from_tokens(neg_tokens, return_pooled=True)
            negative = [[neg_cond, {"pooled_output": neg_pooled}]]
            
            # Encode input to latent
            image_dev = image.to(device)
            input_latent = vae.encode(image_dev[:, :, :, :3])
            noise = comfy.sample.prepare_noise(input_latent, seed)
            
            # Sample with user-controlled denoise
            print(f"[Genfocus Pipeline] Deblur denoise: {deblur_denoise:.3f}")
            sampler = comfy.samplers.KSampler(
                model_deblur,
                steps=deblur_steps,
                device=device,
                sampler=comfy.samplers.sampler_object("euler"),
                scheduler="normal",
                denoise=deblur_denoise,
                model_options={}
            )
            
            deblur_samples = sampler.sample(
                noise, positive, negative,
                cfg=3.5,
                latent_image=input_latent,
                start_step=None, last_step=None,
                force_full_denoise=False,
                denoise_mask=None,
                callback=None,
                disable_pbar=False,
                seed=seed
            )
            
            deblurred = vae.decode(deblur_samples)
            print("[Genfocus Pipeline] Stage 1: Complete")
        
        # If blur_strength is 0, return deblurred result
        if blur_strength == 0:
            print("[Genfocus Pipeline] K=0, returning deblurred result only")
            zeros = torch.zeros(B, original_h, original_w, C, dtype=image.dtype)
            result = deblurred[:, :original_h, :original_w, :]
            return (result, result, zeros, zeros)
        
        # =========== Depth Estimation ===========
        print("[Genfocus Pipeline] Estimating depth...")
        
        depth_model_obj = depth_model["model"]
        transform = depth_model["transform"]
        depth_device = depth_model["device"]
        
        # Convert to PIL and run depth estimation
        deblurred_np = (deblurred[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(deblurred_np)
        
        img_tensor = transform(pil_image)
        if depth_device == "cuda":
            img_tensor = img_tensor.to("cuda")
        
        with torch.no_grad():
            prediction = depth_model_obj.infer(img_tensor, f_px=None)
        
        depth = prediction["depth"].cpu().numpy().squeeze()
        
        # Calculate disparity
        safe_depth = np.where(depth > 0.0, depth, np.finfo(np.float32).max)
        disparity = 1.0 / safe_depth
        
        # Resize disparity to image size if needed
        if disparity.shape[0] != H or disparity.shape[1] != W:
            disp_pil = Image.fromarray(disparity)
            disp_pil = disp_pil.resize((W, H), Image.BILINEAR)
            disparity = np.array(disp_pil)
        
        print("[Genfocus Pipeline] Depth estimation complete")
        
        # =========== Compute Defocus Map ===========
        print("[Genfocus Pipeline] Computing defocus map...")
        
        focus_x = int(W * focus_x_percent / 100.0)
        focus_y = int(H * focus_y_percent / 100.0)
        focus_x = max(0, min(focus_x, W - 1))
        focus_y = max(0, min(focus_y, H - 1))
        
        disp_focus = disparity[focus_y, focus_x]
        dmf = disparity - disp_focus
        defocus_abs = np.abs(blur_strength * dmf)
        
        MAX_COC = 100.0
        defocus_normalized = np.clip(defocus_abs / MAX_COC, 0.0, 1.0)
        dmf_tensor = torch.from_numpy(defocus_normalized.astype(np.float32)).unsqueeze(0).to(device)
        
        print(f"[Genfocus Pipeline] Focus at ({focus_x}, {focus_y})")
        print(f"[Genfocus Pipeline] Defocus range: {defocus_normalized.min():.3f} - {defocus_normalized.max():.3f}")
        
        # =========== STAGE 2: Bokeh ===========
        print("[Genfocus Pipeline] Stage 2: Generating bokeh...")
        
        # Use base model - Genfocus LoRAs require custom transformer_forward
        model_bokeh = model
        if bokeh_lora_path:
            print(f"[Genfocus Pipeline] Note: Using base model (custom LoRA application not yet implemented)")
        
        # Create bokeh conditioning
        bokeh_prompt = "an excellent photo with a large aperture, shallow depth of field, beautiful bokeh"
        tokens = clip.tokenize(bokeh_prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        positive = [[cond, {"pooled_output": pooled}]]
        
        neg_tokens = clip.tokenize("sharp everywhere, deep focus, small aperture")
        neg_cond, neg_pooled = clip.encode_from_tokens(neg_tokens, return_pooled=True)
        negative = [[neg_cond, {"pooled_output": neg_pooled}]]
        
        # Encode deblurred to latent
        deblurred_dev = deblurred.to(device)
        deblur_latent = vae.encode(deblurred_dev[:, :, :, :3])
        
        # Calculate denoise based on average defocus, scaled by user's max
        # Keep denoise LOW to preserve original image - blur is via conditioning
        avg_defocus = defocus_normalized.mean()
        # Scale from 0.15 to user's bokeh_denoise_max based on defocus
        denoise = 0.15 + ((bokeh_denoise_max - 0.15) * min(1.0, avg_defocus))
        print(f"[Genfocus Pipeline] Bokeh denoise: {denoise:.3f} (avg_defocus={avg_defocus:.3f}, max={bokeh_denoise_max:.2f})")
        
        noise = comfy.sample.prepare_noise(deblur_latent, seed + 1)
        
        sampler = comfy.samplers.KSampler(
            model_bokeh,
            steps=bokeh_steps,
            device=device,
            sampler=comfy.samplers.sampler_object("euler"),
            scheduler="normal",
            denoise=denoise,
            model_options={}
        )
        
        bokeh_samples = sampler.sample(
            noise, positive, negative,
            cfg=1.0,  # Lower CFG for bokeh
            latent_image=deblur_latent,
            start_step=None, last_step=None,
            force_full_denoise=False,
            denoise_mask=None,
            callback=None,
            disable_pbar=False,
            seed=seed + 1
        )
        
        # Blend using defocus map: keep sharp where defocus is low
        # Ensure tensors are on the same device
        bokeh_device = bokeh_samples.device
        dmf_latent = F.interpolate(
            dmf_tensor.unsqueeze(1).to(bokeh_device),
            size=(bokeh_samples.shape[2], bokeh_samples.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        deblur_latent = deblur_latent.to(bokeh_device)
        blended_samples = deblur_latent * (1 - dmf_latent) + bokeh_samples * dmf_latent
        
        bokeh_result = vae.decode(blended_samples)
        print("[Genfocus Pipeline] Stage 2: Complete")
        
        # =========== Prepare Outputs ===========
        # Convert depth to display format
        depth_display = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_rgb = np.stack([depth_display, depth_display, depth_display], axis=-1)
        depth_out = torch.from_numpy(depth_rgb.astype(np.float32)).unsqueeze(0)
        
        # Convert defocus map to display format  
        defocus_rgb = np.stack([defocus_normalized, defocus_normalized, defocus_normalized], axis=-1)
        defocus_out = torch.from_numpy(defocus_rgb.astype(np.float32)).unsqueeze(0)
        
        # Crop back to original size
        if new_h != original_h or new_w != original_w:
            bokeh_result = bokeh_result[:, :original_h, :original_w, :]
            deblurred = deblurred[:, :original_h, :original_w, :]
            depth_out = depth_out[:, :original_h, :original_w, :]
            defocus_out = defocus_out[:, :original_h, :original_w, :]
        
        print("[Genfocus Pipeline] Complete!")
        
        return (bokeh_result, deblurred, depth_out, defocus_out)
