"""
Genfocus Model Loader - Loads FluxPipeline with Genfocus LoRAs.

This node loads the FLUX.1-dev model and the DeblurNet/BokehNet LoRA adapters
required for the Genfocus generative refocusing pipeline.

Model folder locations:
  - LoRAs: ComfyUI/models/genfocus/
  - FLUX:  ComfyUI/models/diffusers/FLUX.1-dev/
"""

import os
import torch
import folder_paths


def get_genfocus_lora_files():
    """Get list of available Genfocus LoRA files from models/genfocus folder."""
    try:
        files = folder_paths.get_filename_list("genfocus")
        # Filter to only show safetensors files
        lora_files = [f for f in files if f.endswith('.safetensors')]
        return ["none"] + lora_files if lora_files else ["none"]
    except:
        return ["none"]


def get_diffusers_models():
    """
    Get list of available diffusers-format FLUX models.
    
    Looks in ComfyUI/models/diffusers/ for folders containing model_index.json
    Also includes the HuggingFace repo ID as an option.
    """
    models = []
    
    # Check for local diffusers models first
    diffusers_path = os.path.join(folder_paths.models_dir, "diffusers")
    if os.path.exists(diffusers_path):
        for name in os.listdir(diffusers_path):
            model_path = os.path.join(diffusers_path, name)
            # Check if it looks like a diffusers model (has model_index.json)
            if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "model_index.json")):
                models.append(f"local:{name}")
    
    # Add HuggingFace option (requires huggingface-cli login)
    models.append("hf:black-forest-labs/FLUX.1-dev")
    
    if not models:
        models = ["hf:black-forest-labs/FLUX.1-dev"]
    
    return models


class GenfocusModelLoader:
    """
    Loads the FLUX.1-dev pipeline with Genfocus LoRA adapters.
    
    This node initializes the diffusers FluxPipeline and loads the
    DeblurNet and BokehNet LoRA adapters for multi-conditional generation.
    
    Model locations:
    - FLUX model: Place in ComfyUI/models/diffusers/FLUX.1-dev/ (diffusers format)
    - LoRAs: Place in ComfyUI/models/genfocus/ (safetensors format)
    
    Note: This uses diffusers format. For ComfyUI's native UNET loading,
    use the legacy Genfocus nodes instead (which work with standard KSampler).
    """
    
    CATEGORY = "Genfocus"
    RETURN_TYPES = ("GENFOCUS_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_model"
    
    # Cache for loaded pipeline
    _cached_pipeline = None
    _cached_config = None
    
    @classmethod
    def INPUT_TYPES(cls):
        available_models = get_diffusers_models()
        available_loras = get_genfocus_lora_files()
        
        # Prefer local model if available
        default_model = available_models[0] if available_models else "hf:black-forest-labs/FLUX.1-dev"
        
        # Try to auto-detect deblur/bokeh loras
        deblur_default = "none"
        bokeh_default = "none"
        for lora in available_loras:
            if "deblur" in lora.lower():
                deblur_default = lora
            if "bokeh" in lora.lower():
                bokeh_default = lora
        
        return {
            "required": {
                "flux_model": (available_models, {
                    "default": default_model,
                    "tooltip": "FLUX model: 'local:' = models/diffusers/, 'hf:' = HuggingFace (requires login)"
                }),
                "precision": (["bf16", "fp16", "fp32"], {
                    "default": "bf16",
                    "tooltip": "Model precision (bf16 recommended for FLUX)"
                }),
            },
            "optional": {
                "deblur_lora": (available_loras, {
                    "default": deblur_default,
                    "tooltip": "DeblurNet LoRA (from models/genfocus/)"
                }),
                "bokeh_lora": (available_loras, {
                    "default": bokeh_default,
                    "tooltip": "BokehNet LoRA (from models/genfocus/)"
                }),
                "offload_to_cpu": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Offload model to CPU when not in use (saves VRAM but slower)"
                }),
            }
        }
    
    def load_model(
        self,
        flux_model: str,
        precision: str,
        deblur_lora: str = "none",
        bokeh_lora: str = "none",
        offload_to_cpu: bool = False,
    ):
        """Load the Genfocus pipeline."""
        try:
            from diffusers import FluxPipeline
        except ImportError:
            raise RuntimeError(
                "diffusers library not found. Please install with: "
                "pip install diffusers transformers accelerate peft"
            )
        
        # Check cache
        config = (flux_model, precision, deblur_lora, bokeh_lora, offload_to_cpu)
        if GenfocusModelLoader._cached_pipeline is not None and GenfocusModelLoader._cached_config == config:
            print("[Genfocus] Using cached pipeline")
            return (GenfocusModelLoader._cached_pipeline,)
        
        # Determine dtype
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        dtype = dtype_map[precision]
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[Genfocus] Loading FLUX pipeline: {flux_model}")
        print(f"[Genfocus] Precision: {precision}, Device: {device}")
        
        # Load the pipeline based on prefix
        if flux_model.startswith("local:"):
            # Load from local diffusers folder
            model_name = flux_model.replace("local:", "")
            flux_path = os.path.join(folder_paths.models_dir, "diffusers", model_name)
            if not os.path.exists(flux_path):
                raise RuntimeError(
                    f"Local FLUX model not found at: {flux_path}\n"
                    f"Please download FLUX.1-dev in diffusers format and place it there."
                )
            print(f"[Genfocus] Loading from local path: {flux_path}")
            pipeline = FluxPipeline.from_pretrained(flux_path, torch_dtype=dtype)
        elif flux_model.startswith("hf:"):
            # Load from HuggingFace
            repo_id = flux_model.replace("hf:", "")
            print(f"[Genfocus] Loading from HuggingFace: {repo_id}")
            print("[Genfocus] Note: Requires 'huggingface-cli login' for gated models")
            pipeline = FluxPipeline.from_pretrained(repo_id, torch_dtype=dtype)
        else:
            # Fallback - assume HuggingFace ID
            pipeline = FluxPipeline.from_pretrained(flux_model, torch_dtype=dtype)
        
        # Move to device
        if device == "cuda":
            if offload_to_cpu:
                pipeline.enable_model_cpu_offload()
            else:
                pipeline.to("cuda")
        
        # Load LoRA adapters from genfocus folder
        loras_loaded = []
        
        if deblur_lora and deblur_lora != "none":
            try:
                lora_path = folder_paths.get_full_path("genfocus", deblur_lora)
                pipeline.load_lora_weights(lora_path, adapter_name="deblurring")
                loras_loaded.append("deblurring")
                print(f"[Genfocus] Loaded DeblurNet LoRA: {deblur_lora}")
            except Exception as e:
                print(f"[Genfocus] Warning: Failed to load DeblurNet LoRA: {e}")
        
        if bokeh_lora and bokeh_lora != "none":
            try:
                lora_path = folder_paths.get_full_path("genfocus", bokeh_lora)
                pipeline.load_lora_weights(lora_path, adapter_name="bokeh")
                loras_loaded.append("bokeh")
                print(f"[Genfocus] Loaded BokehNet LoRA: {bokeh_lora}")
            except Exception as e:
                print(f"[Genfocus] Warning: Failed to load BokehNet LoRA: {e}")
        
        # CRITICAL: After loading multiple adapters, set them all as active
        # The specify_lora context manager will control which one is actually used per-branch
        if loras_loaded:
            try:
                pipeline.set_adapters(loras_loaded)
                print(f"[Genfocus] Activated adapters: {loras_loaded}")
            except Exception as e:
                print(f"[Genfocus] Warning: Failed to set adapters: {e}")
        
        # Store pipeline info
        pipeline._genfocus_config = {
            "loras_loaded": loras_loaded,
            "device": device,
            "dtype": dtype,
        }
        
        # Cache the pipeline
        GenfocusModelLoader._cached_pipeline = pipeline
        GenfocusModelLoader._cached_config = config
        
        print(f"[Genfocus] Pipeline ready. LoRAs loaded: {loras_loaded}")
        
        return (pipeline,)


class GenfocusSwitchAdapter:
    """
    Switches the active LoRA adapter on the Genfocus pipeline.
    
    This node follows the original Genfocus approach: unload the current LoRA
    and load the new one fresh. This prevents any interference between adapters.
    
    Use this to switch between 'deblurring' and 'bokeh' modes.
    """
    
    CATEGORY = "Genfocus"
    RETURN_TYPES = ("GENFOCUS_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "switch_adapter"
    
    @classmethod
    def INPUT_TYPES(cls):
        available_loras = get_genfocus_lora_files()
        
        # Try to auto-detect deblur/bokeh loras
        deblur_default = "none"
        bokeh_default = "none"
        for lora in available_loras:
            if "deblur" in lora.lower():
                deblur_default = lora
            if "bokeh" in lora.lower():
                bokeh_default = lora
        
        return {
            "required": {
                "pipeline": ("GENFOCUS_PIPELINE",),
                "adapter_mode": (["deblurring", "bokeh", "none"], {
                    "default": "deblurring",
                    "tooltip": "Which LoRA adapter to use"
                }),
            },
            "optional": {
                "deblur_lora": (available_loras, {
                    "default": deblur_default,
                    "tooltip": "DeblurNet LoRA file"
                }),
                "bokeh_lora": (available_loras, {
                    "default": bokeh_default,
                    "tooltip": "BokehNet LoRA file"
                }),
            }
        }
    
    def switch_adapter(
        self, 
        pipeline, 
        adapter_mode: str,
        deblur_lora: str = "none",
        bokeh_lora: str = "none",
    ):
        """
        Switch the active LoRA adapter using the original Genfocus approach:
        unload current LoRA, then load the new one fresh.
        """
        # Unload any existing LoRA weights
        try:
            pipeline.unload_lora_weights()
            print("[Genfocus] Unloaded existing LoRA weights")
        except Exception as e:
            print(f"[Genfocus] Note: {e}")
        
        if adapter_mode == "none":
            pipeline._genfocus_config["loras_loaded"] = []
            print("[Genfocus] No LoRA active")
            return (pipeline,)
        
        # Load the requested LoRA fresh
        lora_file = deblur_lora if adapter_mode == "deblurring" else bokeh_lora
        
        if lora_file == "none":
            print(f"[Genfocus] Warning: No LoRA file specified for {adapter_mode}")
            return (pipeline,)
        
        try:
            lora_path = folder_paths.get_full_path("genfocus", lora_file)
            pipeline.load_lora_weights(lora_path, adapter_name=adapter_mode)
            pipeline.set_adapters([adapter_mode])
            pipeline._genfocus_config["loras_loaded"] = [adapter_mode]
            print(f"[Genfocus] Loaded and activated: {adapter_mode} from {lora_file}")
        except Exception as e:
            print(f"[Genfocus] Error loading LoRA: {e}")
        
        return (pipeline,)


class GenfocusDeblurLoader:
    """
    Simplified loader for deblur-only workflows.
    
    Loads FLUX.1-dev with only the DeblurNet LoRA. This is more efficient
    for workflows that only need all-in-focus estimation.
    """
    
    CATEGORY = "Genfocus"
    RETURN_TYPES = ("GENFOCUS_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_model"
    
    # Separate cache for deblur-only pipeline
    _cached_pipeline = None
    _cached_config = None
    
    @classmethod
    def INPUT_TYPES(cls):
        available_models = get_diffusers_models()
        available_loras = get_genfocus_lora_files()
        
        default_model = available_models[0] if available_models else "hf:black-forest-labs/FLUX.1-dev"
        
        # Auto-detect deblur lora
        deblur_default = "none"
        for lora in available_loras:
            if "deblur" in lora.lower():
                deblur_default = lora
                break
        
        return {
            "required": {
                "flux_model": (available_models, {
                    "default": default_model,
                }),
                "deblur_lora": (available_loras, {
                    "default": deblur_default,
                    "tooltip": "DeblurNet LoRA"
                }),
            },
            "optional": {
                "precision": (["bf16", "fp16", "fp32"], {
                    "default": "bf16",
                }),
            }
        }
    
    def load_model(
        self,
        flux_model: str,
        deblur_lora: str,
        precision: str = "bf16",
    ):
        """Load FLUX with only the DeblurNet LoRA."""
        try:
            from diffusers import FluxPipeline
        except ImportError:
            raise RuntimeError("diffusers library not found")
        
        # Check cache
        config = (flux_model, deblur_lora, precision)
        if GenfocusDeblurLoader._cached_pipeline is not None and GenfocusDeblurLoader._cached_config == config:
            print("[GenfocusDeblur] Using cached pipeline")
            return (GenfocusDeblurLoader._cached_pipeline,)
        
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        dtype = dtype_map[precision]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[GenfocusDeblur] Loading FLUX: {flux_model}")
        
        # Load pipeline
        if flux_model.startswith("local:"):
            model_name = flux_model.replace("local:", "")
            flux_path = os.path.join(folder_paths.models_dir, "diffusers", model_name)
            pipeline = FluxPipeline.from_pretrained(flux_path, torch_dtype=dtype)
        else:
            repo_id = flux_model.replace("hf:", "") if flux_model.startswith("hf:") else flux_model
            pipeline = FluxPipeline.from_pretrained(repo_id, torch_dtype=dtype)
        
        if device == "cuda":
            pipeline.to("cuda")
        
        # Load ONLY the deblur LoRA
        loras_loaded = []
        if deblur_lora and deblur_lora != "none":
            try:
                lora_path = folder_paths.get_full_path("genfocus", deblur_lora)
                pipeline.load_lora_weights(lora_path, adapter_name="deblurring")
                pipeline.set_adapters(["deblurring"])
                loras_loaded.append("deblurring")
                print(f"[GenfocusDeblur] Loaded DeblurNet: {deblur_lora}")
            except Exception as e:
                print(f"[GenfocusDeblur] Error: {e}")
        
        pipeline._genfocus_config = {
            "loras_loaded": loras_loaded,
            "device": device,
            "dtype": dtype,
        }
        
        GenfocusDeblurLoader._cached_pipeline = pipeline
        GenfocusDeblurLoader._cached_config = config
        
        print(f"[GenfocusDeblur] Ready with LoRA: {loras_loaded}")
        return (pipeline,)


class GenfocusUnloadModels:
    """
    Unload Genfocus pipeline and DepthPro models from VRAM/RAM.
    
    Use this node to free up memory when you're done with Genfocus,
    or when models get orphaned from reloading workflows.
    
    This clears:
    - All cached Genfocus pipelines (from all loader nodes)
    - DepthPro model cache
    - Runs garbage collection and clears CUDA cache
    """
    
    CATEGORY = "Genfocus"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "unload"
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unload_genfocus": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Unload Genfocus FLUX pipeline from memory"
                }),
                "unload_depthpro": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Unload DepthPro depth model from memory"
                }),
            },
            "optional": {
                "trigger": ("*", {"tooltip": "Connect anything here to trigger unload"}),
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always execute when triggered
        import time
        return time.time()
    
    def unload(self, unload_genfocus=True, unload_depthpro=True, trigger=None):
        import gc
        
        status_parts = []
        
        if unload_genfocus:
            # Clear all Genfocus caches
            cleared = []
            
            if GenfocusModelLoader._cached_pipeline is not None:
                del GenfocusModelLoader._cached_pipeline
                GenfocusModelLoader._cached_pipeline = None
                GenfocusModelLoader._cached_config = None
                cleared.append("GenfocusModelLoader")
            
            if GenfocusDeblurLoader._cached_pipeline is not None:
                del GenfocusDeblurLoader._cached_pipeline
                GenfocusDeblurLoader._cached_pipeline = None
                GenfocusDeblurLoader._cached_config = None
                cleared.append("GenfocusDeblurLoader")
            
            if cleared:
                status_parts.append(f"Cleared Genfocus: {', '.join(cleared)}")
                print(f"[Unload] Cleared Genfocus pipelines: {', '.join(cleared)}")
            else:
                status_parts.append("No Genfocus pipelines in cache")
                print("[Unload] No Genfocus pipelines found in cache")
        
        if unload_depthpro:
            # Try to clear DepthPro - it doesn't cache at class level,
            # but we can try to find and clear global references
            try:
                # Try importing the actual depth_pro library and clearing any global state
                import sys
                cleared_dp = False
                
                # Check if depth_pro module has any cached models
                if 'depth_pro' in sys.modules:
                    dp_mod = sys.modules['depth_pro']
                    # Some versions cache the model globally
                    for attr in ['_model', 'model', '_cached_model']:
                        if hasattr(dp_mod, attr):
                            delattr(dp_mod, attr)
                            cleared_dp = True
                
                if cleared_dp:
                    status_parts.append("Cleared DepthPro globals")
                    print("[Unload] Cleared DepthPro global references")
                else:
                    status_parts.append("DepthPro: no cache found (normal)")
                    print("[Unload] DepthPro doesn't use class-level cache - DEPTH_PRO_MODEL outputs hold the reference")
                    print("[Unload] Tip: Disconnect the depth model output to allow garbage collection")
            except Exception as e:
                status_parts.append(f"DepthPro note: {str(e)[:30]}")
                print(f"[Unload] DepthPro info: {e}")
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Report VRAM status
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            status_parts.append(f"VRAM: {allocated:.1f}GB used, {reserved:.1f}GB reserved")
            print(f"[Unload] CUDA cache cleared. VRAM: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
        
        status = " | ".join(status_parts)
        return (status,)


NODE_CLASS_MAPPINGS = {
    "GenfocusModelLoader": GenfocusModelLoader,
    "GenfocusSwitchAdapter": GenfocusSwitchAdapter,
    "GenfocusDeblurLoader": GenfocusDeblurLoader,
    "GenfocusUnloadModels": GenfocusUnloadModels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GenfocusModelLoader": "Genfocus Model Loader",
    "GenfocusSwitchAdapter": "Genfocus Switch Adapter",
    "GenfocusDeblurLoader": "Genfocus Deblur Loader",
    "GenfocusUnloadModels": "Genfocus Unload Models",
}
