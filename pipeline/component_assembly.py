"""
FLUX component assembler - builds a FluxPipeline from individual component files.

Each component (transformer, T5, CLIP-L, VAE) can be supplied as either a single
.safetensors file or a diffusers-format folder. Tokenizers, scheduler, and
VAE/text-encoder configs are auto-downloaded from FLUX.1-schnell on first use
(Apache 2.0, non-gated, ~5MB).

This module has NO ComfyUI dependencies - testable standalone and the designated
extension point for future quantized variants. To add NF4/INT8 support, subclass
FluxComponentAssembler and override `_load_transformer`.

Requires:
    diffusers >= 0.30 (for FluxTransformer2DModel.from_single_file)
    transformers >= 4.40
    huggingface_hub
    safetensors
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Tuple, Union

import torch

# Bumped on every meaningful change so we can verify which version is running.
# If you don't see this version printed in your console on first use, Python
# is loading stale bytecode - delete the pipeline/__pycache__ folder.
_ASSEMBLER_VERSION = "0.4-shape-preflight"

PathLike = Union[str, Path]

# FLUX.1-schnell is Apache-licensed and non-gated. Tokenizers, scheduler, and
# VAE/text-encoder configs are byte-identical (or compatible) with FLUX.1-dev.
_FLUX_REFERENCE_REPO = "black-forest-labs/FLUX.1-schnell"
_REFERENCE_PATTERNS = [
    "tokenizer/*",
    "tokenizer_2/*",
    "scheduler/*",
    "text_encoder/config.json",
    "text_encoder_2/config.json",
    "vae/config.json",
]


def _log(msg: str) -> None:
    """Match the existing [Refocus] / [Genfocus] print-style logging."""
    print(f"[Refocus.Assembly] {msg}")


# Print version on module import so we know which build is loaded.
print(f"[Refocus.Assembly] component_assembly module loaded, version={_ASSEMBLER_VERSION}")


def _is_diffusers_folder(path: Path) -> bool:
    """A folder qualifies as a diffusers component if it has a config.json."""
    return path.is_dir() and (path / "config.json").exists()


def _smart_remap_state_dict(model_keys: set, file_state_dict: dict) -> dict:
    """
    Remap a state dict's keys to match the model's expected key namespace by
    auto-detecting prefix differences.

    Handles three cases:
      A) Direct match (>50% overlap): no remap, return as-is.
      B) File has a prefix wrapping the model keys (e.g. file is
         'text_encoder.text_model.X', model expects 'text_model.X'):
         strip the wrapping prefix.
      C) Model expects a prefix the file doesn't have (e.g. file is
         'embeddings.X', model expects 'text_model.embeddings.X'):
         add the missing prefix.

    Detection works by sampling a longest-suffix match between file and model
    keys, then voting on the most common prefix delta. Quietly returns the
    original dict if no confident remap is found - `load_state_dict(strict=False)`
    will then surface the mismatch.
    """
    file_keys = set(file_state_dict.keys())

    # Case A: meaningful overlap - leave alone
    overlap = len(model_keys & file_keys)
    if overlap > 0 and overlap >= 0.5 * min(len(model_keys), len(file_keys)):
        return file_state_dict

    # Case B: detect a prefix to STRIP from file keys.
    # For each file key, find the model key that's a suffix of it; the leading
    # bit of the file key is the candidate prefix to strip.
    sorted_model = sorted(model_keys, key=len, reverse=True)
    strip_candidates: Counter = Counter()
    for fk in file_keys:
        for mk in sorted_model:
            if fk.endswith(mk) and fk != mk:
                strip_candidates[fk[: -len(mk)]] += 1
                break

    if strip_candidates:
        prefix, hits = strip_candidates.most_common(1)[0]
        # Require the prefix to explain a substantial fraction of file keys
        if hits >= 0.5 * len(file_keys):
            _log(f"Auto-stripping prefix {prefix!r} from {hits}/{len(file_keys)} keys")
            return {
                (k[len(prefix):] if k.startswith(prefix) else k): v
                for k, v in file_state_dict.items()
            }

    # Case C: detect a prefix to ADD to file keys.
    sorted_file = sorted(file_keys, key=len, reverse=True)
    add_candidates: Counter = Counter()
    for mk in model_keys:
        for fk in sorted_file:
            if mk.endswith(fk) and mk != fk:
                add_candidates[mk[: -len(fk)]] += 1
                break

    if add_candidates:
        prefix, hits = add_candidates.most_common(1)[0]
        if hits >= 0.5 * len(model_keys):
            _log(f"Auto-prepending prefix {prefix!r} to {hits}/{len(file_keys)} keys")
            return {(prefix + k): v for k, v in file_state_dict.items()}

    # No confident remap - let the loader handle it
    return file_state_dict


def _validate_shapes(
    model: "torch.nn.Module",
    state_dict: dict,
    model_kind: str,
    file_name: str,
) -> None:
    """
    Pre-flight check: compare tensor shapes between file and model for any
    overlapping keys. Raise an actionable RuntimeError if mismatches are found.

    This produces a much clearer error than torch's default load_state_dict
    error, with model-kind-specific guidance for common failure modes
    (e.g. extended-context CLIP-L variants like clip_l_hidream).
    """
    model_sd = model.state_dict()
    mismatches: List[Tuple[str, tuple, tuple]] = []
    for key, file_tensor in state_dict.items():
        if key not in model_sd:
            continue
        expected = tuple(model_sd[key].shape)
        actual = tuple(file_tensor.shape)
        if expected != actual:
            mismatches.append((key, expected, actual))

    if not mismatches:
        return

    lines = [
        f"{model_kind.upper()} text encoder file '{file_name}' is incompatible "
        f"with FLUX. Found {len(mismatches)} tensor shape mismatch(es):",
        "",
    ]
    for key, expected, actual in mismatches[:5]:
        lines.append(f"  {key}: expected {expected}, got {actual}")
    if len(mismatches) > 5:
        lines.append(f"  ... and {len(mismatches) - 5} more")

    # Targeted guidance for known cases
    lines.append("")
    if model_kind == "clip":
        for key, expected, actual in mismatches:
            if "position_embedding" in key and len(expected) == 2 and len(actual) == 2:
                if expected[0] == 77 and actual[0] != 77:
                    lines.append(
                        f"The position_embedding has {actual[0]} positions but FLUX "
                        f"requires 77. This file is an extended-context CLIP-L "
                        f"variant (e.g. for HiDream-I1), not the standard CLIP-L "
                        f"that FLUX uses."
                    )
                    lines.append(
                        "Fix: pick a vanilla 'clip_l.safetensors' file in the "
                        "clip_l_encoder dropdown (the kind shipped with standard "
                        "ComfyUI FLUX setups)."
                    )
                    break
        else:
            lines.append(
                "This usually means the file is for a different CLIP variant "
                "(CLIP-G, CLIP-Bigg, etc.) or a non-FLUX architecture. Pick a "
                "standard CLIP-L file in the clip_l_encoder dropdown."
            )
    elif model_kind == "t5":
        lines.append(
            "This usually means the T5 file is for a different model size "
            "(t5-base, t5-large, etc.). FLUX requires T5-XXL. Pick a "
            "'t5xxl_*.safetensors' file in the t5_encoder dropdown."
        )

    raise RuntimeError("\n".join(lines))


class FluxComponentAssembler:
    """
    Builds a FluxPipeline from individually-supplied components.

    Designed for extension: subclass and override any `_load_*` method to swap in
    quantized loading (NF4, INT8, fp8) without touching pipeline assembly.

    Args:
        cache_dir: Directory for auto-downloaded reference assets.
            Recommended: <node_package>/.flux_cache/
    """

    def __init__(self, cache_dir: PathLike):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._reference_dir: Path | None = None  # populated lazily

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def assemble(
        self,
        transformer_path: PathLike,
        t5_path: PathLike,
        clip_l_path: PathLike,
        vae_path: PathLike,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        offload_to_cpu: bool = False,
    ):
        """Assemble and return a FluxPipeline placed on the requested device."""
        from diffusers import FluxPipeline

        _log(f"Assembling FLUX pipeline from components (assembler v{_ASSEMBLER_VERSION})")

        transformer = self._load_transformer(Path(transformer_path), dtype)
        text_encoder = self._load_clip_l(Path(clip_l_path), dtype)
        text_encoder_2 = self._load_t5(Path(t5_path), dtype)
        vae = self._load_vae(Path(vae_path), dtype)
        tokenizer, tokenizer_2 = self._load_tokenizers()
        scheduler = self._load_scheduler()

        pipeline = FluxPipeline(
            transformer=transformer,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            vae=vae,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            scheduler=scheduler,
        )

        if device == "cuda" and torch.cuda.is_available():
            if offload_to_cpu:
                pipeline.enable_model_cpu_offload()
            else:
                pipeline.to("cuda")

        _log("Pipeline assembled successfully")
        return pipeline

    # ------------------------------------------------------------------ #
    #  Component loaders - override these for quantized variants
    # ------------------------------------------------------------------ #

    def _load_transformer(self, path: Path, dtype: torch.dtype):
        """Load FluxTransformer2DModel from .safetensors or diffusers folder."""
        from diffusers import FluxTransformer2DModel

        if path.is_file() and path.suffix == ".safetensors":
            _log(f"Loading transformer (single file): {path.name}")
            # diffusers handles BFL->diffusers key remapping internally
            return FluxTransformer2DModel.from_single_file(
                str(path), torch_dtype=dtype
            )
        if _is_diffusers_folder(path):
            _log(f"Loading transformer (diffusers folder): {path.name}")
            return FluxTransformer2DModel.from_pretrained(
                str(path), torch_dtype=dtype
            )
        raise FileNotFoundError(
            f"Transformer not found or unsupported format: {path}\n"
            f"Expected a .safetensors file or a diffusers folder."
        )

    def _load_vae(self, path: Path, dtype: torch.dtype):
        """
        Load AutoencoderKL from .safetensors or diffusers folder.

        IMPORTANT: from_single_file defaults to the SD1.5 VAE config (4-channel
        latent). FLUX uses a 16-channel VAE, so we must pass the FLUX VAE
        config explicitly via the `config` argument.
        """
        from diffusers import AutoencoderKL

        if path.is_file() and path.suffix == ".safetensors":
            flux_vae_config_dir = self._ensure_reference_assets() / "vae"
            _log(f"Loading VAE (single file): {path.name} (config={flux_vae_config_dir})")
            return AutoencoderKL.from_single_file(
                str(path),
                config=str(flux_vae_config_dir),
                torch_dtype=dtype,
            )
        if _is_diffusers_folder(path):
            _log(f"Loading VAE (diffusers folder): {path.name}")
            return AutoencoderKL.from_pretrained(str(path), torch_dtype=dtype)
        raise FileNotFoundError(f"VAE not found: {path}")

    def _load_clip_l(self, path: Path, dtype: torch.dtype):
        """Load CLIP-L text encoder."""
        from transformers import CLIPTextModel

        if _is_diffusers_folder(path):
            _log(f"Loading CLIP-L (folder): {path.name}")
            return CLIPTextModel.from_pretrained(str(path), torch_dtype=dtype)
        if path.is_file() and path.suffix == ".safetensors":
            _log(f"Loading CLIP-L (single file): {path.name}")
            return self._load_text_encoder_from_state_dict(
                path, ref_subfolder="text_encoder", model_kind="clip", dtype=dtype
            )
        raise FileNotFoundError(f"CLIP-L not found: {path}")

    def _load_t5(self, path: Path, dtype: torch.dtype):
        """Load T5-XXL text encoder."""
        from transformers import T5EncoderModel

        if _is_diffusers_folder(path):
            _log(f"Loading T5 (folder): {path.name}")
            return T5EncoderModel.from_pretrained(str(path), torch_dtype=dtype)
        if path.is_file() and path.suffix == ".safetensors":
            _log(f"Loading T5 (single file): {path.name}")
            return self._load_text_encoder_from_state_dict(
                path, ref_subfolder="text_encoder_2", model_kind="t5", dtype=dtype
            )
        raise FileNotFoundError(f"T5 not found: {path}")

    # ------------------------------------------------------------------ #
    #  Shared internals
    # ------------------------------------------------------------------ #

    def _load_text_encoder_from_state_dict(
        self,
        path: Path,
        ref_subfolder: str,
        model_kind: str,
        dtype: torch.dtype,
    ):
        """
        Load a text encoder from a single .safetensors using the reference
        repo's config.

        ComfyUI single-file text encoders use varying key prefixes (raw HF,
        wrapped under 'text_encoder.', 'cond_stage_model.', etc.). We use
        smart prefix-detection to remap keys to the model's expected
        namespace automatically, then pre-flight validate tensor shapes
        before calling load_state_dict so users get an actionable error
        when the file is for a different architecture.
        """
        from safetensors.torch import load_file

        config_dir = self._ensure_reference_assets() / ref_subfolder

        if model_kind == "t5":
            from transformers import T5Config, T5EncoderModel
            config = T5Config.from_pretrained(str(config_dir))
            model = T5EncoderModel(config)
        elif model_kind == "clip":
            from transformers import CLIPTextConfig, CLIPTextModel
            config = CLIPTextConfig.from_pretrained(str(config_dir))
            model = CLIPTextModel(config)
        else:
            raise ValueError(f"Unknown model_kind: {model_kind}")

        state_dict = load_file(str(path))
        model_keys = set(model.state_dict().keys())
        state_dict = _smart_remap_state_dict(model_keys, state_dict)

        # Pre-flight: catch architecture-mismatch errors with clear guidance
        # before falling into torch's cryptic shape error.
        _validate_shapes(model, state_dict, model_kind, path.name)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        # T5 ties encoder.embed_tokens to shared, so missing one of them is OK
        # if the other is present. Filter benign tied-weight misses from the warning.
        if model_kind == "t5":
            tied_keys = {"encoder.embed_tokens.weight"}
            file_provided = set(state_dict.keys())
            if "shared.weight" in file_provided:
                missing = [k for k in missing if k not in tied_keys]

        if missing:
            _log(
                f"WARNING: {model_kind.upper()} {len(missing)} missing keys "
                f"(first 3: {missing[:3]})"
            )
        if unexpected:
            _log(
                f"WARNING: {model_kind.upper()} {len(unexpected)} unexpected keys "
                f"(first 3: {unexpected[:3]})"
            )
        if not missing and not unexpected:
            _log(f"{model_kind.upper()} loaded cleanly")

        return model.to(dtype=dtype)

    def _load_tokenizers(self):
        """Load CLIPTokenizer + T5TokenizerFast from the reference cache."""
        from transformers import CLIPTokenizer, T5TokenizerFast

        ref = self._ensure_reference_assets()
        tokenizer = CLIPTokenizer.from_pretrained(str(ref / "tokenizer"))
        tokenizer_2 = T5TokenizerFast.from_pretrained(str(ref / "tokenizer_2"))
        return tokenizer, tokenizer_2

    def _load_scheduler(self):
        """Load the FlowMatchEulerDiscrete scheduler from the reference cache."""
        from diffusers import FlowMatchEulerDiscreteScheduler

        ref = self._ensure_reference_assets()
        return FlowMatchEulerDiscreteScheduler.from_pretrained(
            str(ref / "scheduler")
        )

    def _ensure_reference_assets(self) -> Path:
        """
        Ensure tokenizers, scheduler, and component configs are cached locally.
        Downloads from FLUX.1-schnell on first call.

        Verifies that all expected files are present before trusting the cache;
        if anything is missing (e.g. cache predates a new pattern), the marker
        is invalidated and assets are re-downloaded.
        """
        if self._reference_dir is not None and self._reference_dir.exists():
            return self._reference_dir

        target = self.cache_dir / "flux_reference"
        marker = target / ".complete"

        # Verify cache has everything we need before trusting the marker.
        # This handles the case where _REFERENCE_PATTERNS was extended after
        # an earlier successful download.
        required_files = [
            "tokenizer/tokenizer_config.json",
            "tokenizer_2/tokenizer_config.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder_2/config.json",
            "vae/config.json",
        ]
        cache_complete = marker.exists() and all(
            (target / rf).exists() for rf in required_files
        )

        if cache_complete:
            self._reference_dir = target
            return target

        if marker.exists():
            _log("Reference cache is missing files - re-downloading")
            try:
                marker.unlink()
            except OSError:
                pass

        _log(
            f"Downloading FLUX tokenizers, scheduler, and component configs "
            f"from {_FLUX_REFERENCE_REPO} (~5MB)"
        )
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                "huggingface_hub is required for first-time component setup. "
                "Install with: pip install huggingface_hub"
            ) from e

        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=_FLUX_REFERENCE_REPO,
            allow_patterns=_REFERENCE_PATTERNS,
            local_dir=str(target),
        )
        marker.touch()
        self._reference_dir = target
        _log(f"Reference assets cached at: {target}")
        return target
