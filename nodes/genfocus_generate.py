"""
Genfocus Generate - Core generation node using the Genfocus pipeline.

This node runs the multi-conditional Genfocus generation with proper
transformer_forward that supports multiple image branches and LoRA switching.
"""

import torch
import numpy as np
from PIL import Image


class GenfocusCondition:
    """
    Creates a Genfocus Condition object from an image or tensor.
    
    Conditions are additional inputs to the generation that can have
    their own LoRA adapter and position embeddings.
    """
    
    CATEGORY = "Genfocus"
    RETURN_TYPES = ("GENFOCUS_CONDITION",)
    RETURN_NAMES = ("condition",)
    FUNCTION = "create_condition"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "adapter": (["deblurring", "bokeh", "none"], {
                    "default": "deblurring",
                    "tooltip": "LoRA adapter to use for this condition"
                }),
            },
            "optional": {
                "position_offset_h": ("INT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "tooltip": "Height offset for position embeddings"
                }),
                "position_offset_w": ("INT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "tooltip": "Width offset for position embeddings"
                }),
                "position_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Scale factor for position embeddings"
                }),
                "no_preprocess": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Skip VAE preprocessing (for tensor inputs like defocus maps)"
                }),
            }
        }
    
    def create_condition(
        self,
        image,
        adapter: str,
        position_offset_h: int = 0,
        position_offset_w: int = 0,
        position_scale: float = 1.0,
        no_preprocess: bool = False,
    ):
        """Create a Genfocus Condition object."""
        from ..pipeline.condition import Condition
        
        # Convert ComfyUI image tensor to PIL
        if isinstance(image, torch.Tensor):
            # ComfyUI format: [B, H, W, C] with values 0-1
            img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)
        else:
            pil_image = image
        
        # Set position delta
        position_delta = None
        if position_offset_h != 0 or position_offset_w != 0:
            position_delta = [position_offset_h, position_offset_w]
        
        # Map adapter name
        adapter_name = adapter if adapter != "none" else None
        
        condition = Condition(
            condition=pil_image,
            adapter_setting=adapter_name,
            position_delta=position_delta,
            position_scale=position_scale,
            no_preprocess=no_preprocess,
        )
        
        return (condition,)


class GenfocusDefocusMapCondition:
    """
    Creates a Genfocus Condition from a defocus map tensor.
    
    This is specifically for the bokeh stage where we pass the
    defocus map (normalized CoC) as a conditioning input.
    
    Connect the defocus_map output from Compute Defocus Map node.
    """
    
    CATEGORY = "Genfocus"
    RETURN_TYPES = ("GENFOCUS_CONDITION",)
    RETURN_NAMES = ("condition",)
    FUNCTION = "create_condition"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "defocus_map": ("IMAGE",),
            },
            "optional": {
                "max_coc": ("FLOAT", {
                    "default": 100.0,
                    "min": 1.0,
                    "max": 500.0,
                    "tooltip": "Maximum circle of confusion for normalization"
                }),
            }
        }
    
    def create_condition(self, defocus_map, max_coc: float = 100.0):
        """Create a condition from a defocus map."""
        from ..pipeline.condition import Condition
        
        # Defocus map should be [B, H, W, C] with values already normalized to 0-1
        # from the ComputeDefocusMap node. max_coc is kept as parameter for backwards
        # compatibility but we DON'T divide again - the map is already normalized.
        if isinstance(defocus_map, torch.Tensor):
            # Already normalized 0-1 from ComputeDefocusMap, just clamp for safety
            normalized = defocus_map.clamp(0, 1)
            
            # Convert to format expected by the pipeline
            # [B, H, W, C] -> [B, C, H, W]
            if normalized.dim() == 4:
                tensor_input = normalized[0].permute(2, 0, 1).unsqueeze(0)
            else:
                tensor_input = normalized
        else:
            tensor_input = defocus_map
        
        # Log the defocus map range for debugging
        print(f"[DefocusCondition] Creating condition with range: {tensor_input.min():.4f} - {tensor_input.max():.4f}")
        
        condition = Condition(
            condition=tensor_input,
            adapter_setting="bokeh",
            position_delta=[0, 0],
            position_scale=1.0,
            no_preprocess=True,  # Don't run through image processor
        )
        
        return (condition,)


class GenfocusGenerate:
    """
    Runs the Genfocus multi-conditional generation.
    
    This is the core generation node that uses the custom transformer_forward
    to process multiple image conditions with per-branch LoRA adapters.
    
    The main_adapter setting controls which LoRA is applied to the main generation
    branch (the output image). This should typically match your task:
    - "deblurring" for all-in-focus restoration
    - "bokeh" for shallow depth-of-field generation
    - "none" if using conditions with their own adapters
    """
    
    CATEGORY = "Genfocus"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    
    # Standard prompts for Genfocus operations - matches original demo.py exactly
    PROMPT_PRESETS = {
        "deblur": "a sharp photo with everything in focus",
        "bokeh": "an excellent photo with a large aperture",
        "custom": "",
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("GENFOCUS_PIPELINE",),
                "prompt_preset": (["deblur", "bokeh", "custom"], {
                    "default": "bokeh",
                    "tooltip": "Preset prompt for operation type"
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 16,
                    "tooltip": "Output width (FLUX native: 1024)"
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 16,
                    "tooltip": "Output height (FLUX native: 1024)"
                }),
                "steps": ("INT", {
                    "default": 28,
                    "min": 1,
                    "max": 100,
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance (1.0 = no CFG)"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**32 - 1,
                }),
            },
            "optional": {
                "custom_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Custom prompt (only used when preset is 'custom')"
                }),
                "condition_1": ("GENFOCUS_CONDITION",),
                "condition_2": ("GENFOCUS_CONDITION",),
                "condition_3": ("GENFOCUS_CONDITION",),
                "latents": ("LATENT",),
                "main_adapter": (["auto", "deblurring", "bokeh", "none"], {
                    "default": "auto",
                    "tooltip": "LoRA for main branch. 'auto' matches prompt_preset"
                }),
                "kv_cache": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable KV caching for efficiency"
                }),
            }
        }
    
    def generate(
        self,
        pipeline,
        prompt_preset: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int,
        custom_prompt: str = "",
        condition_1=None,
        condition_2=None,
        condition_3=None,
        latents=None,
        main_adapter: str = "auto",
        kv_cache: bool = False,
    ):
        """Run Genfocus generation."""
        from ..pipeline.generate import generate, seed_everything
        
        # Resolve prompt from preset
        if prompt_preset == "custom":
            prompt = custom_prompt if custom_prompt else self.PROMPT_PRESETS["bokeh"]
        else:
            prompt = self.PROMPT_PRESETS[prompt_preset]
        
        # Resolve main_adapter from preset if auto
        # NOTE: main_adapter controls which adapter is used for condition branches
        # The main latent branch always uses NO LoRA (None)
        if main_adapter == "auto":
            if prompt_preset == "deblur":
                condition_adapter = "deblurring"
            elif prompt_preset == "bokeh":
                condition_adapter = "bokeh"
            else:
                condition_adapter = None
        elif main_adapter == "none":
            condition_adapter = None
        else:
            condition_adapter = main_adapter
        
        # Activate only the relevant adapter (original demo.py pattern)
        if condition_adapter:
            try:
                pipeline.set_adapters([condition_adapter])
                print(f"[GenfocusGenerate] Activated adapter: {condition_adapter} only")
            except Exception as e:
                print(f"[GenfocusGenerate] Warning: Could not set adapters: {e}")
        
        # Set seed
        seed_everything(seed)
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        
        # Gather conditions
        conditions = []
        for cond in [condition_1, condition_2, condition_3]:
            if cond is not None:
                conditions.append(cond)
        
        print(f"[Genfocus Generate] Prompt: '{prompt[:50]}...'")
        print(f"[Genfocus Generate] Condition adapter: {condition_adapter}")
        # Prepare latents if provided
        input_latents = None
        if latents is not None:
            input_latents = latents.get("samples", None)
        
        print(f"[Genfocus Generate] Running with {len(conditions)} conditions")
        print(f"[Genfocus Generate] Size: {width}x{height}, Steps: {steps}, Guidance: {guidance_scale}")
        
        # Run generation
        # NOTE: main_adapter=None means main branch has no LoRA
        # Conditions will use their own adapters from the Condition objects
        result = generate(
            pipeline,
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            latents=input_latents,
            conditions=conditions,
            main_adapter=None,  # Original doesn't use LoRA on main branch
            kv_cache=kv_cache,
        )
        
        # Convert output to ComfyUI format
        if hasattr(result, 'images'):
            images = result.images
        else:
            images = result[0]
        
        # Handle PIL images
        if isinstance(images[0], Image.Image):
            output_tensors = []
            for img in images:
                np_img = np.array(img).astype(np.float32) / 255.0
                output_tensors.append(torch.from_numpy(np_img))
            output = torch.stack(output_tensors)
        else:
            output = images
        
        return (output,)


class GenfocusDeblur:
    """
    Convenience node for the deblurring stage of Genfocus.
    
    Takes a blurry input image and generates a sharp, all-in-focus result.
    Uses the standard deblur prompt automatically.
    """
    
    CATEGORY = "Genfocus"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("deblurred",)
    FUNCTION = "deblur"
    
    # Standard deblur prompt - matches original demo.py exactly
    DEFAULT_PROMPT = "a sharp photo with everything in focus"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("GENFOCUS_PIPELINE",),
                "image": ("IMAGE",),
                "steps": ("INT", {
                    "default": 28,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Denoising steps (28 recommended)"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**32 - 1,
                }),
            },
            "optional": {
                "use_custom_prompt": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use custom prompt instead of standard deblur prompt"
                }),
                "custom_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Custom prompt (only used if use_custom_prompt is True)"
                }),
            }
        }
    
    def deblur(
        self,
        pipeline,
        image,
        steps: int,
        seed: int,
        use_custom_prompt: bool = False,
        custom_prompt: str = "",
    ):
        """Run the deblurring stage."""
        from ..pipeline.generate import generate, seed_everything
        from ..pipeline.condition import Condition
        
        # Select prompt
        prompt = custom_prompt if use_custom_prompt and custom_prompt else self.DEFAULT_PROMPT
        
        # Get image dimensions from ComfyUI tensor [B, H, W, C]
        B, H, W, C = image.shape
        
        # CRITICAL: Align dimensions to multiples of 16 (required by FLUX VAE)
        aligned_W = ((W + 15) // 16) * 16
        aligned_H = ((H + 15) // 16) * 16
        
        # Convert to PIL
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)
        
        # Resize if needed to align dimensions
        if aligned_W != W or aligned_H != H:
            print(f"[Genfocus Deblur] Aligning {W}x{H} → {aligned_W}x{aligned_H} (multiple of 16)")
            pil_image = pil_image.resize((aligned_W, aligned_H), Image.LANCZOS)
            W, H = aligned_W, aligned_H
        
        print(f"[Genfocus Deblur] Processing {W}x{H} image, {steps} steps")
        
        # Create conditions as per Genfocus demo
        # Condition 0: Black image with position offset (for structure)
        black_image = Image.new("RGB", (W, H), (0, 0, 0))
        cond_0 = Condition(black_image, "deblurring", [0, 32], 1.0)
        
        # Condition 1: Input image (for content)
        cond_1 = Condition(pil_image, "deblurring", [0, 0], 1.0)
        
        # Set seed and run
        seed_everything(seed)
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        
        # CRITICAL: Original demo.py only activates ONE LoRA at a time for deblur
        # Set ONLY deblurring adapter as active (not bokeh)
        try:
            pipeline.set_adapters(["deblurring"])
            print(f"[Genfocus Deblur] Activated adapter: deblurring only")
        except Exception as e:
            print(f"[Genfocus Deblur] Warning: Could not set adapters: {e}")
        
        # NOTE: Original Genfocus does NOT pass main_adapter for deblur stage
        # The main latent branch uses NO LoRA - only the conditions use the LoRA
        # Original demo.py uses default guidance_scale=3.5 for deblurring
        result = generate(
            pipeline,
            prompt=prompt,
            height=H,
            width=W,
            num_inference_steps=steps,
            guidance_scale=3.5,  # Original uses default 3.5 for deblurring
            generator=generator,
            conditions=[cond_0, cond_1],
            main_adapter=None,  # Original doesn't use LoRA on main branch
        )
        
        # Convert output
        if hasattr(result, 'images'):
            images = result.images
        else:
            images = result[0]
        
        if isinstance(images[0], Image.Image):
            np_img = np.array(images[0]).astype(np.float32) / 255.0
            output = torch.from_numpy(np_img).unsqueeze(0)
        else:
            output = images
        
        return (output,)


class GenfocusBokeh:
    """
    Convenience node for the bokeh stage of Genfocus.
    
    Takes a sharp image and defocus map, and generates realistic bokeh.
    Uses the standard bokeh prompt automatically.
    """
    
    CATEGORY = "Genfocus"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("bokeh_result",)
    FUNCTION = "apply_bokeh"
    
    # Standard bokeh prompt - matches original demo.py exactly
    DEFAULT_PROMPT = "an excellent photo with a large aperture"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("GENFOCUS_PIPELINE",),
                "image": ("IMAGE",),
                "defocus_map": ("IMAGE",),
                "steps": ("INT", {
                    "default": 28,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Denoising steps (28 recommended)"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**32 - 1,
                }),
            },
            "optional": {
                "use_custom_prompt": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use custom prompt instead of standard bokeh prompt"
                }),
                "custom_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Custom prompt (only used if use_custom_prompt is True)"
                }),
            }
        }
    
    def apply_bokeh(
        self,
        pipeline,
        image,
        defocus_map,
        steps: int,
        seed: int,
        use_custom_prompt: bool = False,
        custom_prompt: str = "",
    ):
        """Run the bokeh stage."""
        from ..pipeline.generate import generate, seed_everything
        from ..pipeline.condition import Condition
        
        # Select prompt
        prompt = custom_prompt if use_custom_prompt and custom_prompt else self.DEFAULT_PROMPT
        
        B, H, W, C = image.shape
        dm_B, dm_H, dm_W, dm_C = defocus_map.shape
        
        # Check if defocus map matches image dimensions
        if dm_H != H or dm_W != W:
            print(f"[Genfocus Bokeh] WARNING: Defocus map ({dm_W}x{dm_H}) doesn't match image ({W}x{H})!")
            print(f"[Genfocus Bokeh] Resizing defocus map to match image...")
            defocus_map = torch.nn.functional.interpolate(
                defocus_map.permute(0, 3, 1, 2),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
        
        # CRITICAL: Align dimensions to multiples of 16 (required by FLUX VAE)
        aligned_W = ((W + 15) // 16) * 16
        aligned_H = ((H + 15) // 16) * 16
        
        # Convert image to PIL
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)
        
        # Resize if needed to align dimensions
        if aligned_W != W or aligned_H != H:
            print(f"[Genfocus Bokeh] Aligning {W}x{H} → {aligned_W}x{aligned_H} (multiple of 16)")
            pil_image = pil_image.resize((aligned_W, aligned_H), Image.LANCZOS)
            # Also resize defocus map
            defocus_map = torch.nn.functional.interpolate(
                defocus_map.permute(0, 3, 1, 2),
                size=(aligned_H, aligned_W),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
            W, H = aligned_W, aligned_H
        
        print(f"[Genfocus Bokeh] Processing {W}x{H} image, {steps} steps")
        
        # Defocus map from Compute Defocus Map is already normalized 0-1
        # Convert to [B, C, H, W] format for the pipeline
        defocus_tensor = defocus_map[0].permute(2, 0, 1).unsqueeze(0)
        
        # Create conditions
        cond_img = Condition(pil_image, "bokeh")
        cond_dmf = Condition(defocus_tensor, "bokeh", [0, 0], 1.0, no_preprocess=True)
        
        # Set seed and run
        seed_everything(seed)
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        
        # CRITICAL: Original demo.py only activates ONE LoRA at a time for bokeh
        # Set ONLY bokeh adapter as active (not deblurring)
        try:
            pipeline.set_adapters(["bokeh"])
            print(f"[Genfocus Bokeh] Activated adapter: bokeh only")
        except Exception as e:
            print(f"[Genfocus Bokeh] Warning: Could not set adapters: {e}")
        
        # NOTE: Original Genfocus does NOT pass main_adapter for bokeh stage
        # The main latent branch uses NO LoRA - only the conditions use the LoRA
        result = generate(
            pipeline,
            prompt=prompt,
            height=H,
            width=W,
            num_inference_steps=steps,
            guidance_scale=3.5,  # Original uses default 3.5
            generator=generator,
            conditions=[cond_img, cond_dmf],
            main_adapter=None,  # Original doesn't use LoRA on main branch
        )
        
        # Convert output
        if hasattr(result, 'images'):
            images = result.images
        else:
            images = result[0]
        
        if isinstance(images[0], Image.Image):
            np_img = np.array(images[0]).astype(np.float32) / 255.0
            output = torch.from_numpy(np_img).unsqueeze(0)
        else:
            output = images
        
        return (output,)


NODE_CLASS_MAPPINGS = {
    "GenfocusCondition": GenfocusCondition,
    "GenfocusDefocusMapCondition": GenfocusDefocusMapCondition,
    "GenfocusGenerate": GenfocusGenerate,
    "GenfocusDeblur": GenfocusDeblur,
    "GenfocusBokeh": GenfocusBokeh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GenfocusCondition": "Genfocus Condition",
    "GenfocusDefocusMapCondition": "Genfocus Defocus Map Condition",
    "GenfocusGenerate": "Genfocus Generate",
    "GenfocusDeblur": "Genfocus Deblur",
    "GenfocusBokeh": "Genfocus Bokeh",
}
