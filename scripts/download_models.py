#!/usr/bin/env python3
"""
Download script for Genfocus models.

Downloads FLUX.1-dev (diffusers format) and Genfocus LoRAs to the correct
ComfyUI model folders.

Usage:
    python download_models.py                    # Download all models
    python download_models.py --flux-only        # Only download FLUX.1-dev
    python download_models.py --loras-only       # Only download LoRAs
    python download_models.py --check            # Check what's already downloaded
"""

import os
import sys
import argparse
from pathlib import Path

# Find ComfyUI models directory
def find_models_dir():
    """Find the ComfyUI models directory."""
    # Check relative to this script
    script_dir = Path(__file__).parent.parent
    
    # Look for models dir relative to custom_nodes
    possible_paths = [
        script_dir.parent.parent / "models",  # ComfyUI/custom_nodes/Refocus -> ComfyUI/models
        Path.cwd() / "models",
        Path.cwd().parent / "models",
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # Default: create relative to script
    default = script_dir.parent.parent / "models"
    print(f"[INFO] Models directory not found, will create: {default}")
    return default


def check_hf_login():
    """Check if user is logged in to HuggingFace."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        print(f"[OK] Logged in to HuggingFace as: {user['name']}")
        return True
    except Exception as e:
        print(f"[WARNING] Not logged in to HuggingFace: {e}")
        print("\nTo download FLUX.1-dev (gated model), you need to:")
        print("  1. Create account at https://huggingface.co/join")
        print("  2. Accept FLUX.1-dev license at https://huggingface.co/black-forest-labs/FLUX.1-dev")
        print("  3. Run: huggingface-cli login")
        return False


def download_flux_model(models_dir: Path, precision: str = "bf16"):
    """Download FLUX.1-dev in diffusers format."""
    try:
        from diffusers import FluxPipeline
        import torch
    except ImportError:
        print("[ERROR] diffusers not installed. Run: pip install diffusers transformers accelerate")
        return False
    
    # Check login first
    if not check_hf_login():
        print("\n[ERROR] Please login to HuggingFace first to download FLUX.1-dev")
        return False
    
    diffusers_dir = models_dir / "diffusers"
    flux_dir = diffusers_dir / "FLUX.1-dev"
    
    if flux_dir.exists() and (flux_dir / "model_index.json").exists():
        print(f"[SKIP] FLUX.1-dev already exists at: {flux_dir}")
        return True
    
    print(f"\n[DOWNLOAD] Downloading FLUX.1-dev to: {flux_dir}")
    print("[INFO] This is a large model (~23GB), please be patient...")
    
    # Create directory
    diffusers_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine dtype
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map.get(precision, torch.bfloat16)
    
    try:
        # Download and save locally
        print("[INFO] Downloading from HuggingFace (this may take a while)...")
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=dtype,
        )
        
        print(f"[INFO] Saving to: {flux_dir}")
        pipeline.save_pretrained(flux_dir)
        
        # Clean up to free memory
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"[OK] FLUX.1-dev downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to download FLUX.1-dev: {e}")
        return False


def download_genfocus_loras(models_dir: Path):
    """Download Genfocus LoRA files."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    
    genfocus_dir = models_dir / "genfocus"
    genfocus_dir.mkdir(parents=True, exist_ok=True)
    
    loras = [
        ("deblurNet.safetensors", "DeblurNet LoRA"),
        ("bokehNet.safetensors", "BokehNet LoRA"),
    ]
    
    repo_id = "rayray9999/Genfocus"
    success = True
    
    for filename, description in loras:
        local_path = genfocus_dir / filename
        
        if local_path.exists():
            print(f"[SKIP] {description} already exists: {local_path}")
            continue
        
        print(f"[DOWNLOAD] Downloading {description}...")
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=genfocus_dir,
                local_dir_use_symlinks=False,
            )
            print(f"[OK] {description} downloaded: {downloaded}")
        except Exception as e:
            print(f"[ERROR] Failed to download {description}: {e}")
            success = False
    
    return success


def download_depth_pro(models_dir: Path):
    """Download DepthPro weights."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    
    checkpoints_dir = models_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    local_path = checkpoints_dir / "depth_pro.pt"
    
    if local_path.exists():
        print(f"[SKIP] DepthPro already exists: {local_path}")
        return True
    
    print("[DOWNLOAD] Downloading DepthPro weights (~500MB)...")
    try:
        # Try the official Apple repo first
        downloaded = hf_hub_download(
            repo_id="apple/DepthPro",
            filename="depth_pro.pt",
            local_dir=checkpoints_dir,
            local_dir_use_symlinks=False,
        )
        print(f"[OK] DepthPro downloaded: {downloaded}")
        return True
    except Exception as e:
        print(f"[WARNING] Could not download from apple/DepthPro: {e}")
        print("[INFO] You may need to download manually from:")
        print("       https://huggingface.co/apple/DepthPro")
        return False


def check_status(models_dir: Path):
    """Check what models are already downloaded."""
    print("\n=== Model Status ===\n")
    
    # Check FLUX
    flux_dir = models_dir / "diffusers" / "FLUX.1-dev"
    if flux_dir.exists() and (flux_dir / "model_index.json").exists():
        print(f"✅ FLUX.1-dev: {flux_dir}")
    else:
        print(f"❌ FLUX.1-dev: Not found (expected at {flux_dir})")
    
    # Check LoRAs
    genfocus_dir = models_dir / "genfocus"
    for lora in ["deblurNet.safetensors", "bokehNet.safetensors"]:
        lora_path = genfocus_dir / lora
        if lora_path.exists():
            size_mb = lora_path.stat().st_size / (1024 * 1024)
            print(f"✅ {lora}: {size_mb:.1f} MB")
        else:
            print(f"❌ {lora}: Not found")
    
    # Check DepthPro
    depth_pro = models_dir / "checkpoints" / "depth_pro.pt"
    if depth_pro.exists():
        size_mb = depth_pro.stat().st_size / (1024 * 1024)
        print(f"✅ depth_pro.pt: {size_mb:.1f} MB")
    else:
        print(f"⚠️  depth_pro.pt: Not found (optional)")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download Genfocus models for ComfyUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_models.py                  # Download everything
  python download_models.py --flux-only      # Only FLUX.1-dev
  python download_models.py --loras-only     # Only LoRAs
  python download_models.py --check          # Check status
  python download_models.py --precision fp16 # Use fp16 for FLUX
        """
    )
    
    parser.add_argument("--flux-only", action="store_true", help="Only download FLUX.1-dev")
    parser.add_argument("--loras-only", action="store_true", help="Only download Genfocus LoRAs")
    parser.add_argument("--depth-pro", action="store_true", help="Also download DepthPro weights")
    parser.add_argument("--check", action="store_true", help="Check what's already downloaded")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16",
                        help="Precision for FLUX model (default: bf16)")
    parser.add_argument("--models-dir", type=Path, help="Override models directory path")
    
    args = parser.parse_args()
    
    # Find models directory
    if args.models_dir:
        models_dir = args.models_dir
    else:
        models_dir = find_models_dir()
    
    print(f"[INFO] Models directory: {models_dir}")
    
    # Check status only
    if args.check:
        check_status(models_dir)
        return
    
    # Download based on flags
    success = True
    
    if args.flux_only:
        success = download_flux_model(models_dir, args.precision)
    elif args.loras_only:
        success = download_genfocus_loras(models_dir)
    else:
        # Download all
        print("\n=== Downloading Genfocus LoRAs ===")
        if not download_genfocus_loras(models_dir):
            success = False
        
        print("\n=== Downloading FLUX.1-dev ===")
        if not download_flux_model(models_dir, args.precision):
            success = False
        
        if args.depth_pro:
            print("\n=== Downloading DepthPro ===")
            if not download_depth_pro(models_dir):
                success = False
    
    # Final status
    print("\n" + "="*50)
    check_status(models_dir)
    
    if success:
        print("[OK] All downloads completed successfully!")
    else:
        print("[WARNING] Some downloads failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
