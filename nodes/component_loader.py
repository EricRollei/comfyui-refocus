"""
Genfocus Component Loader - assembles a FluxPipeline from individual component
files (transformer, T5, CLIP-L, VAE), then applies the Genfocus LoRAs.

Designed for users who have FLUX as a single-file Civitai-style checkpoint and
the standard ComfyUI text encoders / VAE - without needing the full ~23GB
diffusers FLUX folder.

The output type matches the existing GenfocusModelLoader: a FluxPipeline with a
`_genfocus_config` dict attached. All downstream Genfocus nodes work unchanged.

NOTE: Genfocus LoRAs were trained on FLUX.1-dev. They will load on any
FLUX-dev-architecture transformer (including fine-tunes), but quality with
heavily-stylized fine-tunes is untested - your mileage may vary.
"""

import os
from pathlib import Path
from typing import List, Optional

import torch
import folder_paths

# Folder roots scanned for each component. Listed in priority order - earlier
# entries appear first in the dropdown. Folders that aren't registered in this
# ComfyUI install are silently skipped.
_TRANSFORMER_FOLDERS = ("checkpoints", "unet", "diffusion_models")
_T5_FOLDERS          = ("text_encoders", "clip")
_CLIP_L_FOLDERS      = ("text_encoders", "clip")
_VAE_FOLDERS         = ("vae",)

# Where to cache auto-downloaded tokenizers + scheduler config (~5MB, one-time)
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_FLUX_CACHE_DIR = _PACKAGE_ROOT / ".flux_cache"


# --------------------------------------------------------------------------- #
#  Dropdown listing helpers
# --------------------------------------------------------------------------- #

def _list_safetensors(folder_types: tuple) -> List[str]:
    """
    Return entries of the form "folder_type:filename" for every .safetensors
    file in any of the given folder types. Skips folder types that aren't
    registered. De-duplicates by (folder_type, filename) tuple.
    """
    seen = set()
    out: List[str] = []
    for ft in folder_types:
        try:
            files = folder_paths.get_filename_list(ft)
        except Exception:
            continue
        for f in files:
            if not f.endswith(".safetensors"):
                continue
            key = (ft, f)
            if key in seen:
                continue
            seen.add(key)
            out.append(f"{ft}:{f}")
    return out or ["none"]


def _resolve_entry(entry: str) -> Optional[Path]:
    """Resolve a 'folder_type:filename' dropdown entry to an absolute path."""
    if not entry or entry == "none" or ":" not in entry:
        return None
    folder_type, name = entry.split(":", 1)
    full = folder_paths.get_full_path(folder_type, name)
    return Path(full) if full else None


def _list_genfocus_loras() -> List[str]:
    """Match the helper in genfocus_loader.py."""
    try:
        files = folder_paths.get_filename_list("genfocus")
        loras = [f for f in files if f.endswith(".safetensors")]
        return ["none"] + loras if loras else ["none"]
    except Exception:
        return ["none"]


def _autodetect_lora(loras: List[str], keyword: str) -> str:
    """Find a LoRA whose filename contains `keyword` (case-insensitive)."""
    for lora in loras:
        if keyword in lora.lower():
            return lora
    return "none"


# --------------------------------------------------------------------------- #
#  The node
# --------------------------------------------------------------------------- #

class GenfocusComponentLoader:
    """
    Loads FLUX from individual component files and applies Genfocus LoRAs.

    Use this loader when you have a single-file FLUX checkpoint (Civitai-style)
    and the standard ComfyUI text encoders + VAE, instead of the full diffusers
    FLUX folder.
    """

    CATEGORY = "Genfocus"
    RETURN_TYPES = ("GENFOCUS_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_model"

    # Class-level cache, matching the pattern in GenfocusModelLoader
    _cached_pipeline = None
    _cached_config = None

    @classmethod
    def INPUT_TYPES(cls):
        transformers = _list_safetensors(_TRANSFORMER_FOLDERS)
        t5s          = _list_safetensors(_T5_FOLDERS)
        clip_ls      = _list_safetensors(_CLIP_L_FOLDERS)
        vaes         = _list_safetensors(_VAE_FOLDERS)
        loras        = _list_genfocus_loras()

        deblur_default = _autodetect_lora(loras, "deblur")
        bokeh_default  = _autodetect_lora(loras, "bokeh")

        # Sensible default picks: prefer entries with FLUX/t5/clip/ae in the name
        def _prefer(entries: List[str], keywords: tuple) -> str:
            for e in entries:
                lower = e.lower()
                if any(k in lower for k in keywords):
                    return e
            return entries[0]

        transformer_default = _prefer(transformers, ("flux",))
        t5_default          = _prefer(t5s, ("t5xxl", "t5_xxl", "t5-xxl"))
        clip_l_default      = _prefer(clip_ls, ("clip_l", "clip-l", "clipl"))
        vae_default         = _prefer(vaes, ("ae", "flux"))

        return {
            "required": {
                "transformer": (transformers, {
                    "default": transformer_default,
                    "tooltip": (
                        "FLUX transformer .safetensors. Looks in checkpoints/, unet/, "
                        "diffusion_models/. Format: 'folder:filename'. "
                        "Genfocus LoRAs were trained on FLUX.1-dev - fine-tunes will "
                        "load but quality is untested. Do not use FLUX.1-schnell "
                        "(timestep distillation differs)."
                    ),
                }),
                "t5_encoder": (t5s, {
                    "default": t5_default,
                    "tooltip": (
                        "T5-XXL text encoder .safetensors. Use bf16 or fp16 variants "
                        "(e.g. t5xxl_fp16.safetensors). fp8 variants are not currently "
                        "supported by this loader."
                    ),
                }),
                "clip_l_encoder": (clip_ls, {
                    "default": clip_l_default,
                    "tooltip": "CLIP-L text encoder .safetensors (e.g. clip_l.safetensors)",
                }),
                "vae": (vaes, {
                    "default": vae_default,
                    "tooltip": "FLUX VAE .safetensors (e.g. ae.safetensors)",
                }),
                "precision": (["bf16", "fp16", "fp32"], {
                    "default": "bf16",
                    "tooltip": "Pipeline precision (bf16 recommended for FLUX)",
                }),
            },
            "optional": {
                "deblur_lora": (loras, {
                    "default": deblur_default,
                    "tooltip": "DeblurNet LoRA from models/genfocus/",
                }),
                "bokeh_lora": (loras, {
                    "default": bokeh_default,
                    "tooltip": "BokehNet LoRA from models/genfocus/",
                }),
                "offload_to_cpu": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable CPU offloading to save VRAM (slower)",
                }),
            },
        }

    # ------------------------------------------------------------------ #
    #  Loading
    # ------------------------------------------------------------------ #

    def load_model(
        self,
        transformer: str,
        t5_encoder: str,
        clip_l_encoder: str,
        vae: str,
        precision: str,
        deblur_lora: str = "none",
        bokeh_lora: str = "none",
        offload_to_cpu: bool = False,
    ):
        # Cache check
        config = (transformer, t5_encoder, clip_l_encoder, vae,
                  precision, deblur_lora, bokeh_lora, offload_to_cpu)
        if (GenfocusComponentLoader._cached_pipeline is not None
                and GenfocusComponentLoader._cached_config == config):
            print("[Genfocus.Components] Using cached pipeline")
            return (GenfocusComponentLoader._cached_pipeline,)

        # Resolve dropdown entries to absolute paths
        paths = {
            "transformer":    _resolve_entry(transformer),
            "t5_encoder":     _resolve_entry(t5_encoder),
            "clip_l_encoder": _resolve_entry(clip_l_encoder),
            "vae":            _resolve_entry(vae),
        }
        for label, p in paths.items():
            if p is None:
                raise RuntimeError(
                    f"Could not resolve {label} entry: '{locals()[label] if label in locals() else ''}'\n"
                    f"Make sure the file exists in the expected folder."
                )

        # Build the assembler. Imported lazily so a missing diffusers/transformers
        # install fails on use, not on package import.
        try:
            from ..pipeline.component_assembly import FluxComponentAssembler
        except ImportError as e:
            raise RuntimeError(
                "Failed to import the FLUX component assembler. Ensure diffusers, "
                "transformers, huggingface_hub, and safetensors are installed:\n"
                "  pip install diffusers transformers huggingface_hub safetensors peft accelerate"
            ) from e

        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        dtype = dtype_map[precision]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[Genfocus.Components] Assembling pipeline (precision={precision}, "
              f"device={device}, offload={offload_to_cpu})")

        assembler = FluxComponentAssembler(cache_dir=_FLUX_CACHE_DIR)
        pipeline = assembler.assemble(
            transformer_path=paths["transformer"],
            t5_path=paths["t5_encoder"],
            clip_l_path=paths["clip_l_encoder"],
            vae_path=paths["vae"],
            dtype=dtype,
            device=device,
            offload_to_cpu=offload_to_cpu,
        )

        # Apply Genfocus LoRAs - same code path as GenfocusModelLoader
        loras_loaded = self._apply_loras(pipeline, deblur_lora, bokeh_lora)

        # Attach the same metadata shape downstream nodes expect
        pipeline._genfocus_config = {
            "loras_loaded": loras_loaded,
            "device": device,
            "dtype": dtype,
        }

        # Cache
        GenfocusComponentLoader._cached_pipeline = pipeline
        GenfocusComponentLoader._cached_config = config

        print(f"[Genfocus.Components] Ready. LoRAs loaded: {loras_loaded}")
        return (pipeline,)

    @staticmethod
    def _apply_loras(pipeline, deblur_lora: str, bokeh_lora: str) -> List[str]:
        """Load Genfocus LoRAs onto the pipeline. Mirrors GenfocusModelLoader."""
        loras_loaded: List[str] = []

        if deblur_lora and deblur_lora != "none":
            try:
                lora_path = folder_paths.get_full_path("genfocus", deblur_lora)
                pipeline.load_lora_weights(lora_path, adapter_name="deblurring")
                loras_loaded.append("deblurring")
                print(f"[Genfocus.Components] Loaded DeblurNet LoRA: {deblur_lora}")
            except Exception as e:
                print(f"[Genfocus.Components] WARNING: DeblurNet LoRA failed: {e}")

        if bokeh_lora and bokeh_lora != "none":
            try:
                lora_path = folder_paths.get_full_path("genfocus", bokeh_lora)
                pipeline.load_lora_weights(lora_path, adapter_name="bokeh")
                loras_loaded.append("bokeh")
                print(f"[Genfocus.Components] Loaded BokehNet LoRA: {bokeh_lora}")
            except Exception as e:
                print(f"[Genfocus.Components] WARNING: BokehNet LoRA failed: {e}")

        if loras_loaded:
            try:
                pipeline.set_adapters(loras_loaded)
                print(f"[Genfocus.Components] Activated adapters: {loras_loaded}")
            except Exception as e:
                print(f"[Genfocus.Components] WARNING: set_adapters failed: {e}")

        return loras_loaded


NODE_CLASS_MAPPINGS = {
    "GenfocusComponentLoader": GenfocusComponentLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GenfocusComponentLoader": "Genfocus Model Loader (Components)",
}
