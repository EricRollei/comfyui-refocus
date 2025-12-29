"""
DepthPro nodes for Apple's ml-depth-pro model.

These nodes provide standalone depth estimation that can be used
for Genfocus bokeh generation or any other depth-related tasks.

License Note: Apple ml-depth-pro has its own license terms.
See https://github.com/apple/ml-depth-pro/blob/main/LICENSE
"""

import torch
import numpy as np
from PIL import Image
import folder_paths
import os


class DepthProModelLoader:
    """
    Loads the Apple ml-depth-pro model for high-quality metric depth estimation.
    
    The model provides sharp monocular depth with absolute scale,
    without requiring camera intrinsics.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Use standard ComfyUI checkpoint list - works with all configured paths
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("checkpoints"), {
                    "tooltip": "Select depth_pro.pt model from checkpoints"
                }),
            },
            "optional": {
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to load the model on"
                }),
                "precision": (["float16", "float32"], {
                    "default": "float32",
                    "tooltip": "Model precision. float32 gives best quality, float16 uses less VRAM."
                }),
            }
        }
    
    RETURN_TYPES = ("DEPTH_PRO_MODEL",)
    RETURN_NAMES = ("depth_model",)
    FUNCTION = "load_model"
    CATEGORY = "Refocus/Depth"
    DESCRIPTION = "Load Apple's ml-depth-pro model for metric depth estimation"
    
    def load_model(self, model_name, device="auto", precision="float32"):
        try:
            import depth_pro
            from depth_pro.depth_pro import DepthProConfig
        except ImportError:
            raise ImportError(
                "depth_pro not installed. Please install it with:\n"
                "pip install git+https://github.com/apple/ml-depth-pro.git --no-deps"
            )
        
        # Resolve device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Resolve precision
        torch_precision = torch.float16 if precision == "float16" else torch.float32
        
        # Use standard ComfyUI path resolution
        full_path = folder_paths.get_full_path("checkpoints", model_name)
        
        if full_path is None or not os.path.exists(full_path):
            raise FileNotFoundError(
                f"Could not find {model_name} in configured checkpoint folders"
            )
        
        print(f"[DepthPro] Loading model from: {full_path}")
        print(f"[DepthPro] Device: {device}, Precision: {precision}")
        
        # Create config with the correct checkpoint path
        config = DepthProConfig(
            patch_encoder_preset="dinov2l16_384",
            image_encoder_preset="dinov2l16_384",
            decoder_features=256,
            checkpoint_uri=full_path,
            fov_encoder_preset="dinov2l16_384",
            use_fov_head=True
        )
        
        # Create model and transforms with custom config
        model, transform = depth_pro.create_model_and_transforms(
            config=config,
            device=torch.device(device),
            precision=torch_precision
        )
        model.eval()
        
        print(f"[DepthPro] Model loaded successfully (img_size={model.img_size})")
        
        return ({
            "model": model,
            "transform": transform,
            "device": device,
            "precision": precision
        },)


class DepthProEstimate:
    """
    Estimate depth from an image using DepthPro.
    
    Returns both depth map (in meters) and disparity (1/depth).
    The disparity is useful for defocus map calculation.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_model": ("DEPTH_PRO_MODEL",),
                "image": ("IMAGE",),
            },
            "optional": {
                "focal_length_px": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10000.0,
                    "step": 0.1,
                    "tooltip": "Focal length in pixels. 0 = auto-estimate from FOV head"
                }),
                "depth_mode": (["metric", "relative"], {
                    "default": "relative",
                    "tooltip": "metric = absolute depth in meters (uses focal length). relative = normalized depth structure (ignores focal length, similar to Marigold)."
                }),
                "colormap": (["grayscale", "turbo", "viridis", "plasma", "magma"], {
                    "default": "turbo",
                    "tooltip": "Colormap for visualization. 'turbo' matches Apple's official examples."
                }),
                "interpolation": (["bilinear", "bicubic"], {
                    "default": "bicubic",
                    "tooltip": "Interpolation mode for resizing. 'bicubic' gives sharper edges."
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "FLOAT")
    RETURN_NAMES = ("depth_map", "disparity_map", "raw_depth", "raw_inverse", "visualization", "focal_length")
    FUNCTION = "estimate"
    CATEGORY = "Refocus/Depth"
    DESCRIPTION = "Estimate depth using DepthPro. 'relative' mode for visual quality (like Marigold), 'metric' for absolute depth in meters. raw_depth/raw_inverse are unnormalized for precise measurements."
    
    def estimate(self, depth_model, image, focal_length_px=0.0, depth_mode="relative", colormap="turbo", interpolation="bicubic"):
        import torch.nn as nn
        
        model = depth_model["model"]
        transform = depth_model["transform"]
        device = depth_model["device"]
        
        # Convert ComfyUI image (B,H,W,C) to numpy array (what depth_pro.load_rgb returns)
        # Take first image in batch
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        
        H, W = img_np.shape[:2]
        print(f"[DepthPro] Input image: {W}x{H}, mode={depth_mode}")
        print(f"[DepthPro] Input numpy: shape={img_np.shape}, dtype={img_np.dtype}, range={img_np.min()}-{img_np.max()}")
        
        # Apply transform - expects numpy array HWC uint8
        img_tensor = transform(img_np)
        print(f"[DepthPro] After transform: shape={img_tensor.shape}, dtype={img_tensor.dtype}, range={img_tensor.min().item():.3f}-{img_tensor.max().item():.3f}")
        
        # Ensure tensor is on the correct device
        target_device = next(model.parameters()).device
        if img_tensor.device != target_device:
            img_tensor = img_tensor.to(target_device)
        
        # Add batch dimension if needed
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        print(f"[DepthPro] Final input tensor: {img_tensor.shape}")
        
        with torch.no_grad():
            _, _, tensor_H, tensor_W = img_tensor.shape
            
            if depth_mode == "relative":
                # RELATIVE MODE: Use raw canonical inverse depth (ignores focal length)
                # This gives better visual structure like Marigold
                
                # Resize to model size
                img_size = model.img_size
                resize_needed = tensor_H != img_size or tensor_W != img_size
                print(f"[DepthPro] Model img_size={img_size}, resize_needed={resize_needed}")
                
                if resize_needed:
                    x = nn.functional.interpolate(
                        img_tensor, size=(img_size, img_size), 
                        mode=interpolation, align_corners=False
                    )
                else:
                    x = img_tensor
                
                print(f"[DepthPro] Input to model.forward: {x.shape}")
                
                # Get canonical inverse depth (relative depth, not scaled by focal length)
                canonical_inverse_depth, fov_deg = model.forward(x)
                
                print(f"[DepthPro] Raw canonical_inverse_depth: shape={canonical_inverse_depth.shape}, range={canonical_inverse_depth.min().item():.4f}-{canonical_inverse_depth.max().item():.4f}, std={canonical_inverse_depth.std().item():.4f}")
                
                # Resize back to original resolution
                if resize_needed:
                    canonical_inverse_depth = nn.functional.interpolate(
                        canonical_inverse_depth, size=(tensor_H, tensor_W),
                        mode=interpolation, align_corners=False
                    )
                
                # Use inverse depth directly for visualization (near = high, far = low)
                inverse_depth = canonical_inverse_depth.cpu().numpy().squeeze()
                
                print(f"[DepthPro] inverse_depth numpy: shape={inverse_depth.shape}, range={inverse_depth.min():.4f}-{inverse_depth.max():.4f}")
                
                # Compute depth from inverse (for reference)
                depth = 1.0 / np.clip(inverse_depth, 1e-4, 1e4)
                
                # Estimate focal length for output (but don't use it for depth)
                if fov_deg is not None:
                    f_px = 0.5 * tensor_W / torch.tan(0.5 * torch.deg2rad(fov_deg.to(torch.float)))
                    estimated_focal = f_px.item()
                else:
                    estimated_focal = 0.0
                
                print(f"[DepthPro] Canonical inverse depth range: {inverse_depth.min():.3f} - {inverse_depth.max():.3f}")
                print(f"[DepthPro] Estimated FOV: {fov_deg.item():.1f}° (f_px={estimated_focal:.0f})")
                
            else:
                # METRIC MODE: Use full inference with focal length scaling
                f_px = focal_length_px if focal_length_px > 0 else None
                prediction = model.infer(img_tensor, f_px=f_px, interpolation_mode=interpolation)
                
                depth = prediction["depth"].cpu().numpy().squeeze()
                estimated_focal = prediction.get("focallength_px", focal_length_px)
                if isinstance(estimated_focal, torch.Tensor):
                    estimated_focal = estimated_focal.item()
                
                # Calculate inverse depth from metric depth
                inverse_depth = 1.0 / np.clip(depth, 1e-4, 1e4)
                
                print(f"[DepthPro] Metric depth range: {depth.min():.3f}m - {depth.max():.3f}m")
                print(f"[DepthPro] Focal length: {estimated_focal:.1f}px")
        
        # Normalize for visualization
        # inverse_depth: higher = closer, normalize to 0-1
        inv_min, inv_max = inverse_depth.min(), inverse_depth.max()
        inverse_depth_normalized = np.clip(
            (inverse_depth - inv_min) / (inv_max - inv_min + 1e-8),
            0.0, 1.0
        )
        
        # depth: lower = closer, invert for visualization (near = bright)
        depth_min, depth_max = depth.min(), depth.max()
        depth_normalized = 1.0 - ((depth - depth_min) / (depth_max - depth_min + 1e-8))
        
        print(f"[DepthPro] Output shape: {depth.shape}")
        
        # Convert to ComfyUI format (B,H,W,C)
        def to_comfy_image(arr, use_colormap=True):
            arr = arr.astype(np.float32)
            arr = np.clip(arr, 0.0, 1.0)
            
            if use_colormap and colormap != "grayscale":
                # Use matplotlib colormap for beautiful visualization
                import matplotlib.pyplot as plt
                cmap = plt.get_cmap(colormap)
                arr_rgb = cmap(arr)[..., :3]  # Get RGB, discard alpha
                arr_rgb = arr_rgb.astype(np.float32)
            else:
                # Grayscale - stack to RGB
                arr_rgb = np.stack([arr, arr, arr], axis=-1)
            
            return torch.from_numpy(arr_rgb).unsqueeze(0)
        
        # Create visualization panel (input + depth + histogram like the test script)
        def create_visualization_panel(img_np, inverse_depth, inverse_depth_normalized):
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Input image
            axes[0].imshow(img_np)
            axes[0].set_title('Input')
            axes[0].axis('on')
            
            # Depth map with colorbar
            im = axes[1].imshow(inverse_depth, cmap='turbo')
            axes[1].set_title(f'Canonical Inverse Depth\nRange: {inverse_depth.min():.3f} - {inverse_depth.max():.3f}')
            axes[1].axis('on')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Histogram
            axes[2].hist(inverse_depth.flatten(), bins=100, color='steelblue', edgecolor='none')
            axes[2].set_title('Depth Value Distribution')
            axes[2].set_xlabel('Inverse Depth')
            axes[2].set_ylabel('Frequency')
            
            plt.tight_layout()
            
            # Convert figure to image (modern matplotlib approach)
            fig.canvas.draw()
            vis_np = np.array(fig.canvas.buffer_rgba())[:, :, :3]  # RGBA -> RGB
            plt.close(fig)
            
            # Convert to ComfyUI format (B,H,W,C) normalized 0-1
            vis_tensor = torch.from_numpy(vis_np.astype(np.float32) / 255.0).unsqueeze(0)
            return vis_tensor
        
        # depth_out: normalized depth (near = bright/warm)
        # disparity_out: normalized inverse depth (near = bright/warm)
        depth_out = to_comfy_image(depth_normalized, use_colormap=True)
        disparity_out = to_comfy_image(inverse_depth_normalized, use_colormap=True)
        visualization_out = create_visualization_panel(img_np, inverse_depth, inverse_depth_normalized)
        
        # Raw outputs (unnormalized, for precise measurements)
        # raw_depth: actual depth values (meters in metric mode)
        # raw_inverse: actual inverse depth values
        raw_depth_out = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
        raw_inverse_out = torch.from_numpy(inverse_depth.astype(np.float32)).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
        
        return (depth_out, disparity_out, raw_depth_out, raw_inverse_out, visualization_out, float(estimated_focal))


class DepthMetricToRelative:
    """
    Convert metric depth to relative depth (0-1 range).
    
    Works with any depth map source (DepthPro, Marigold, Midas, etc.).
    Useful for ControlNet compatibility and visualization.
    
    Options:
    - per_image: Normalize each image independently (for batches)
    - invert: Flip near/far (some models output inverted depth)
    - gamma: Bias brightness (>1 = brighter midtones, <1 = darker)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth": ("IMAGE",),
            },
            "optional": {
                "per_image": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Normalize each image in batch independently"
                }),
                "invert": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Invert so near=bright, far=dark (standard for ControlNet)"
                }),
                "gamma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Gamma correction: >1 = brighter midtones, <1 = darker"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("relative_depth",)
    FUNCTION = "convert"
    CATEGORY = "Refocus/Depth"
    DESCRIPTION = "Convert metric/raw depth to relative 0-1 range. Works with any depth source. Use gamma to adjust brightness bias."
    
    def convert(self, depth, per_image=True, invert=True, gamma=1.0):
        # Work on a copy
        relative = depth.detach().clone()
        
        # Use first channel only (depth maps are grayscale)
        if relative.shape[-1] == 3:
            relative = relative[..., 0:1]
        
        # Convert metric depth to relative using 1/(1+d) for better distribution
        relative = 1.0 / (1.0 + relative)
        
        if per_image:
            # Normalize each image independently
            for i in range(relative.shape[0]):
                r_min = relative[i].min()
                r_max = relative[i].max()
                if r_max > r_min:
                    relative[i] = (relative[i] - r_min) / (r_max - r_min)
        else:
            # Normalize across entire batch
            r_min = relative.min()
            r_max = relative.max()
            if r_max > r_min:
                relative = (relative - r_min) / (r_max - r_min)
        
        if not invert:
            relative = 1.0 - relative
        
        if gamma != 1.0:
            relative = relative ** (1.0 / gamma)
        
        # Expand back to 3 channels
        relative = relative.repeat(1, 1, 1, 3)
        
        return (relative,)


class DepthMetricToInverse:
    """
    Convert metric depth to inverse depth (disparity-like).
    
    Simple conversion: output = 1 / (1 + depth)
    
    This preserves the depth structure while compressing far distances.
    Useful for defocus map calculation and visualization.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("inverse_depth",)
    FUNCTION = "convert"
    CATEGORY = "Refocus/Depth"
    DESCRIPTION = "Convert metric depth to inverse depth: 1/(1+d). Compresses far distances, useful for defocus maps."
    
    def convert(self, depth):
        inverse = 1.0 / (1.0 + depth.detach().clone())
        return (inverse,)


class FocalPXtoMM:
    """
    Convert focal length from pixels to millimeters.
    
    Formula: focal_mm = focal_px * sensor_mm / sensor_px
    
    Where sensor_px is the larger dimension of the image (width or height).
    Default sensor_mm (24) assumes a common ~24mm sensor height (full frame).
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "focal_px": ("FLOAT", {
                    "default": 1000.0,
                    "min": 0.01,
                    "max": 100000.0,
                    "step": 0.01,
                    "tooltip": "Focal length in pixels (from DepthPro)"
                }),
                "sensor_mm": ("FLOAT", {
                    "default": 24.0,
                    "min": 0.001,
                    "max": 100.0,
                    "step": 0.001,
                    "tooltip": "Sensor size in mm (longer edge). Common: 24mm (full frame height), 36mm (full frame width)"
                }),
                "image_width": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 16384,
                    "tooltip": "Image width in pixels"
                }),
                "image_height": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 16384,
                    "tooltip": "Image height in pixels"
                }),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("focal_mm", "focal_str")
    FUNCTION = "convert"
    CATEGORY = "Refocus/Depth"
    DESCRIPTION = "Convert focal length from pixels to millimeters. Useful for 3D camera setup."
    
    def convert(self, focal_px, sensor_mm, image_width, image_height):
        sensor_px = max(image_width, image_height)
        focal_mm = focal_px * sensor_mm / sensor_px
        focal_str = f"{focal_mm:.2f}mm"
        print(f"[FocalConvert] {focal_px:.1f}px -> {focal_mm:.2f}mm (sensor: {sensor_mm}mm, {sensor_px}px)")
        return (focal_mm, focal_str)


class FocalMMtoPX:
    """
    Convert focal length from millimeters to pixels.
    
    Formula: focal_px = focal_mm * sensor_px / sensor_mm
    
    Where sensor_px is the larger dimension of the image (width or height).
    Useful when you know the real camera focal length and want to use it
    with DepthPro for accurate metric depth.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "focal_mm": ("FLOAT", {
                    "default": 50.0,
                    "min": 0.01,
                    "max": 2000.0,
                    "step": 0.01,
                    "tooltip": "Focal length in millimeters (from camera/EXIF)"
                }),
                "sensor_mm": ("FLOAT", {
                    "default": 24.0,
                    "min": 0.001,
                    "max": 100.0,
                    "step": 0.001,
                    "tooltip": "Sensor size in mm (longer edge)"
                }),
                "image_width": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 16384,
                    "tooltip": "Image width in pixels"
                }),
                "image_height": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 16384,
                    "tooltip": "Image height in pixels"
                }),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("focal_px", "focal_str")
    FUNCTION = "convert"
    CATEGORY = "Refocus/Depth"
    DESCRIPTION = "Convert focal length from millimeters to pixels. Use result as focal_length_px input to DepthPro."
    
    def convert(self, focal_mm, sensor_mm, image_width, image_height):
        sensor_px = max(image_width, image_height)
        focal_px = focal_mm * sensor_px / sensor_mm
        focal_str = f"{focal_px:.1f}px"
        print(f"[FocalConvert] {focal_mm:.2f}mm -> {focal_px:.1f}px (sensor: {sensor_mm}mm, {sensor_px}px)")
        return (focal_px, focal_str)
