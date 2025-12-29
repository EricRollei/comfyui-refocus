"""
Genfocus LoRA loader node.

Loads the DeblurNet and BokehNet LoRA adapters for use with FLUX models.
These LoRAs are trained on FLUX.1-dev but may work with compatible fine-tunes.

License: Apache 2.0 (same as Genfocus project)
"""

import torch
import os
import folder_paths
from safetensors.torch import load_file


def get_lora_list():
    """Get list of available LoRA files from ComfyUI's loras folders."""
    loras = []
    loras_dirs = folder_paths.get_folder_paths("loras")
    for lora_dir in loras_dirs:
        if os.path.exists(lora_dir):
            for root, dirs, files in os.walk(lora_dir):
                for f in files:
                    if f.endswith(".safetensors"):
                        rel_path = os.path.relpath(os.path.join(root, f), lora_dir)
                        if rel_path not in loras:
                            loras.append(rel_path)
    return sorted(loras) if loras else ["deblurNet.safetensors", "bokehNet.safetensors"]


class GenfocusLoRALoader:
    """
    Load Genfocus LoRA weights (DeblurNet and/or BokehNet).
    
    These are FLUX LoRA adapters specifically trained for:
    - DeblurNet: Restoring all-in-focus images from blurry inputs
    - BokehNet: Generating realistic bokeh/depth-of-field effects
    
    Downloads available from: https://huggingface.co/nycu-cplab/Genfocus-Model
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        loras = get_lora_list()
        
        # Try to find genfocus-specific loras as defaults
        deblur_default = next((l for l in loras if "deblur" in l.lower()), loras[0] if loras else "deblurNet.safetensors")
        bokeh_default = next((l for l in loras if "bokeh" in l.lower()), loras[0] if loras else "bokehNet.safetensors")
        
        return {
            "required": {
                "deblur_lora": (loras, {
                    "default": deblur_default,
                    "tooltip": "Select deblurNet.safetensors"
                }),
                "bokeh_lora": (loras, {
                    "default": bokeh_default,
                    "tooltip": "Select bokehNet.safetensors"
                }),
            },
            "optional": {
                "load_deblur": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Load the DeblurNet LoRA"
                }),
                "load_bokeh": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Load the BokehNet LoRA"
                }),
            }
        }
    
    RETURN_TYPES = ("GENFOCUS_LORAS",)
    RETURN_NAMES = ("genfocus_loras",)
    FUNCTION = "load_loras"
    CATEGORY = "Refocus/LoRA"
    DESCRIPTION = "Load Genfocus DeblurNet and BokehNet LoRA weights"
    
    def _resolve_path(self, lora_name):
        """Resolve a LoRA path using ComfyUI's loras folders."""
        loras_dirs = folder_paths.get_folder_paths("loras")
        for lora_dir in loras_dirs:
            full_path = os.path.join(lora_dir, lora_name)
            if os.path.exists(full_path):
                return full_path
        return None
    
    def load_loras(self, deblur_lora, bokeh_lora, load_deblur=True, load_bokeh=True):
        loras = {
            "deblur": None,
            "deblur_path": None,
            "bokeh": None,
            "bokeh_path": None,
        }
        
        if load_deblur:
            path = self._resolve_path(deblur_lora)
            if path is None:
                print(f"[Genfocus] Warning: DeblurNet LoRA not found: {deblur_lora}")
            else:
                print(f"[Genfocus] Loading DeblurNet from: {path}")
                loras["deblur"] = load_file(path)
                loras["deblur_path"] = path
                print(f"[Genfocus] DeblurNet loaded: {len(loras['deblur'])} tensors")
        
        if load_bokeh:
            path = self._resolve_path(bokeh_lora)
            if path is None:
                print(f"[Genfocus] Warning: BokehNet LoRA not found: {bokeh_lora}")
            else:
                print(f"[Genfocus] Loading BokehNet from: {path}")
                loras["bokeh"] = load_file(path)
                loras["bokeh_path"] = path
                print(f"[Genfocus] BokehNet loaded: {len(loras['bokeh'])} tensors")
        
        return (loras,)
