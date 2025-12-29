"""
Refocus - ComfyUI Nodes for Generative Refocusing

Based on the Genfocus project (https://github.com/rayray9999/Genfocus)
Provides deblurring and bokeh generation using FLUX LoRA adapters.

Two implementations available:
1. Legacy nodes (KSampler-based) - simpler but without proper LoRA conditioning
2. Native Genfocus nodes (diffusers-based) - full multi-conditional support

License: Apache 2.0 (for Genfocus components)
Note: Apple ml-depth-pro has its own license terms
Note: FLUX.1-dev requires acceptance of its license

=== MODEL FOLDER SETUP ===

Genfocus LoRAs:
  Place DeblurNet and BokehNet LoRAs in:
    ComfyUI/models/genfocus/
  
  Expected files:
    - deblurNet.safetensors  (or similar name)
    - bokehNet.safetensors   (or similar name)
  
  Download from: https://huggingface.co/rayray9999/Genfocus

FLUX Diffusers Model:
  Place FLUX.1-dev in diffusers format in:
    ComfyUI/models/diffusers/FLUX.1-dev/
  
  Or use the HuggingFace ID (requires login):
    black-forest-labs/FLUX.1-dev
  
  The folder should contain model_index.json and model files.
"""

import os
import folder_paths

# =============================================================================
# REGISTER CUSTOM MODEL FOLDERS
# =============================================================================

# Register genfocus folder for LoRAs
if "genfocus" not in folder_paths.folder_names_and_paths:
    genfocus_path = os.path.join(folder_paths.models_dir, "genfocus")
    os.makedirs(genfocus_path, exist_ok=True)
    folder_paths.folder_names_and_paths["genfocus"] = ([genfocus_path], {".safetensors", ".pt", ".pth", ".bin"})
    print(f"[Refocus] Genfocus LoRA folder: {genfocus_path}")

# Register diffusers folder for FLUX models  
if "diffusers" not in folder_paths.folder_names_and_paths:
    diffusers_path = os.path.join(folder_paths.models_dir, "diffusers")
    os.makedirs(diffusers_path, exist_ok=True)
    folder_paths.folder_names_and_paths["diffusers"] = ([diffusers_path], set())
    print(f"[Refocus] Diffusers model folder: {diffusers_path}")

# =============================================================================
# IMPORT NODES
# =============================================================================

# Legacy nodes (KSampler-based, work without diffusers)
from .nodes.depth_pro import (
    DepthProModelLoader, 
    DepthProEstimate,
    DepthMetricToRelative,
    DepthMetricToInverse,
    FocalPXtoMM,
    FocalMMtoPX,
)
from .nodes.genfocus_lora import GenfocusLoRALoader
from .nodes.deblur import DeblurNetApply
from .nodes.defocus_map import ComputeDefocusMap, SelectFocusPoint, FocusPointFromMask
from .nodes.bokeh import BokehNetApply
from .nodes.pipeline import GenfocusPipeline

# Native Genfocus nodes (diffusers-based, full implementation)
try:
    from .nodes.genfocus_loader import GenfocusModelLoader, GenfocusSwitchAdapter, GenfocusUnloadModels
    from .nodes.genfocus_generate import (
        GenfocusCondition,
        GenfocusDefocusMapCondition,
        GenfocusGenerate,
        GenfocusDeblur,
        GenfocusBokeh,
    )
    NATIVE_GENFOCUS_AVAILABLE = True
except ImportError as e:
    print(f"[Refocus] Native Genfocus nodes not available: {e}")
    print("[Refocus] Install diffusers, peft, accelerate for full Genfocus support")
    NATIVE_GENFOCUS_AVAILABLE = False

NODE_CLASS_MAPPINGS = {
    # Depth Estimation (Standalone utility)
    "DepthProModelLoader": DepthProModelLoader,
    "DepthProEstimate": DepthProEstimate,
    
    # Depth Utility Nodes
    "DepthMetricToRelative": DepthMetricToRelative,
    "DepthMetricToInverse": DepthMetricToInverse,
    "FocalPXtoMM": FocalPXtoMM,
    "FocalMMtoPX": FocalMMtoPX,
    
    # LoRA Management (Legacy)
    "GenfocusLoRALoader": GenfocusLoRALoader,
    
    # Deblurring Stage (Legacy)
    "DeblurNetApply": DeblurNetApply,
    
    # Defocus Map Generation
    "ComputeDefocusMap": ComputeDefocusMap,
    "SelectFocusPoint": SelectFocusPoint,
    "FocusPointFromMask": FocusPointFromMask,
    
    # Bokeh Generation Stage (Legacy)
    "BokehNetApply": BokehNetApply,
    
    # Full Pipeline (Legacy - Convenience)
    "GenfocusPipeline": GenfocusPipeline,
}

# Add native Genfocus nodes if available
if NATIVE_GENFOCUS_AVAILABLE:
    NODE_CLASS_MAPPINGS.update({
        # Model Loading (Native)
        "GenfocusModelLoader": GenfocusModelLoader,
        "GenfocusSwitchAdapter": GenfocusSwitchAdapter,
        "GenfocusUnloadModels": GenfocusUnloadModels,
        
        # Condition Creation (Native)
        "GenfocusCondition": GenfocusCondition,
        "GenfocusDefocusMapCondition": GenfocusDefocusMapCondition,
        
        # Generation (Native)
        "GenfocusGenerate": GenfocusGenerate,
        "GenfocusDeblurNative": GenfocusDeblur,
        "GenfocusBokehNative": GenfocusBokeh,
    })

NODE_DISPLAY_NAME_MAPPINGS = {
    # Depth Estimation
    "DepthProModelLoader": "Load DepthPro Model",
    "DepthProEstimate": "DepthPro Estimate",
    
    # Depth Utilities
    "DepthMetricToRelative": "Depth Metric to Relative",
    "DepthMetricToInverse": "Depth Metric to Inverse",
    "FocalPXtoMM": "Focal Length PX to MM",
    "FocalMMtoPX": "Focal Length MM to PX",
    
    # LoRA Management (Legacy)
    "GenfocusLoRALoader": "Load Genfocus LoRAs (Legacy)",
    
    # Deblurring (Legacy)
    "DeblurNetApply": "Apply DeblurNet (Legacy)",
    
    # Defocus Map
    "ComputeDefocusMap": "Compute Defocus Map",
    "SelectFocusPoint": "Select Focus Point",
    "FocusPointFromMask": "Focus Point From Mask",
    
    # Bokeh (Legacy)
    "BokehNetApply": "Apply BokehNet (Legacy)",
    
    # Pipeline (Legacy)
    "GenfocusPipeline": "Genfocus Pipeline (Legacy)",
}

# Add native Genfocus display names if available
if NATIVE_GENFOCUS_AVAILABLE:
    NODE_DISPLAY_NAME_MAPPINGS.update({
        # Model Loading (Native)
        "GenfocusModelLoader": "Genfocus Model Loader (Native)",
        "GenfocusSwitchAdapter": "Genfocus Switch Adapter",
        "GenfocusUnloadModels": "Genfocus Unload Models",
        
        # Condition Creation (Native)
        "GenfocusCondition": "Genfocus Condition",
        "GenfocusDefocusMapCondition": "Genfocus Defocus Map Condition",
        
        # Generation (Native)
        "GenfocusGenerate": "Genfocus Generate (Native)",
        "GenfocusDeblurNative": "Genfocus Deblur (Native)",
        "GenfocusBokehNative": "Genfocus Bokeh (Native)",
    })

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
