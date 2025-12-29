"""
Defocus map computation nodes.

These nodes compute the defocus map (DMF) from a depth/disparity map
and a focus point. The defocus map controls bokeh intensity.

License: Apache 2.0 (same as Genfocus project)
"""

import torch
import numpy as np
from PIL import Image


class SelectFocusPoint:
    """
    Select a focus point on an image.
    
    This node allows specifying where the focus should be in the image.
    Objects at this point will remain sharp, while objects at different
    depths will be blurred based on their distance from the focus plane.
    
    ENHANCED: Now accepts an optional MASK input for interactive selection!
    Use with Impact Pack's SAM Detector, MaskPainter, or Mask Rect Area
    to click/draw the focus area. The centroid of the mask becomes the focus point.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Optional mask from SAM/MaskPainter - centroid becomes focus point"
                }),
                "x_percent": ("FLOAT", {
                    "default": 50.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "X coordinate as percentage of image width (ignored if mask provided)"
                }),
                "y_percent": ("FLOAT", {
                    "default": 50.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Y coordinate as percentage of image height (ignored if mask provided)"
                }),
            }
        }
    
    RETURN_TYPES = ("FOCUS_POINT", "IMAGE")
    RETURN_NAMES = ("focus_point", "preview")
    FUNCTION = "select_focus"
    CATEGORY = "Refocus/Defocus"
    DESCRIPTION = "Select focus point on image. Connect a MASK from SAM/MaskPainter for interactive selection, or use percentage sliders."
    
    def _get_mask_centroid(self, mask):
        """Calculate the centroid of the mask (weighted center of mass)."""
        mask_np = mask.cpu().numpy()
        
        # Handle batch dimension
        if mask_np.ndim == 3:
            mask_np = mask_np[0]  # Use first mask in batch
        
        # Find non-zero pixels
        ys, xs = np.where(mask_np > 0.5)
        
        if len(xs) == 0:
            return None, None  # Empty mask
        
        # Calculate centroid (weighted by mask values for soft masks)
        weights = mask_np[ys, xs]
        total_weight = weights.sum()
        
        if total_weight == 0:
            return None, None
        
        centroid_x = int(np.round((xs * weights).sum() / total_weight))
        centroid_y = int(np.round((ys * weights).sum() / total_weight))
        
        return centroid_x, centroid_y
    
    def select_focus(self, image, mask=None, x_percent=50.0, y_percent=50.0):
        B, H, W, C = image.shape
        
        source = "percentage"
        
        # Priority: mask > percentage
        if mask is not None:
            centroid_x, centroid_y = self._get_mask_centroid(mask)
            if centroid_x is not None:
                focus_x = centroid_x
                focus_y = centroid_y
                source = "mask centroid"
            else:
                # Fallback to percentage if mask is empty
                focus_x = int(W * x_percent / 100.0)
                focus_y = int(H * y_percent / 100.0)
                print("[FocusPoint] Warning: Mask is empty, falling back to percentage")
        else:
            focus_x = int(W * x_percent / 100.0)
            focus_y = int(H * y_percent / 100.0)
        
        # Clamp to image bounds
        focus_x = max(0, min(focus_x, W - 1))
        focus_y = max(0, min(focus_y, H - 1))
        
        focus_point = {
            "x": focus_x,
            "y": focus_y,
            "width": W,
            "height": H,
        }
        
        # Create preview with focus point marked
        preview = image.clone()
        
        # Draw a red crosshair at focus point
        r = 10  # Radius of crosshair
        line_width = 2
        for i in range(B):
            # Horizontal line
            y_start = max(0, focus_y - line_width // 2)
            y_end = min(H, focus_y + line_width // 2 + 1)
            x_start = max(0, focus_x - r)
            x_end = min(W, focus_x + r + 1)
            preview[i, y_start:y_end, x_start:x_end, 0] = 1.0  # Red
            preview[i, y_start:y_end, x_start:x_end, 1] = 0.0  # Green
            preview[i, y_start:y_end, x_start:x_end, 2] = 0.0  # Blue
            
            # Vertical line
            y_start = max(0, focus_y - r)
            y_end = min(H, focus_y + r + 1)
            x_start = max(0, focus_x - line_width // 2)
            x_end = min(W, focus_x + line_width // 2 + 1)
            preview[i, y_start:y_end, x_start:x_end, 0] = 1.0
            preview[i, y_start:y_end, x_start:x_end, 1] = 0.0
            preview[i, y_start:y_end, x_start:x_end, 2] = 0.0
        
        print(f"[FocusPoint] Selected via {source}: ({focus_x}, {focus_y}) in {W}x{H} image")
        
        return (focus_point, preview)


class ComputeDefocusMap:
    """
    Compute the defocus map from depth/disparity and focus point.
    
    The defocus map (DMF) indicates how much blur each pixel should receive
    based on its distance from the focus plane.
    
    Formula: dmf = K * |disparity - disparity_at_focus|
    
    Where K is the blur strength parameter.
    
    NOTE: The original Genfocus uses raw (unnormalized) defocus maps where values
    often don't span the full 0-1 range. This is by design - the network was trained
    this way. However, you can enable 'normalize_for_genfocus' to stretch the values
    to full 0-1 range for potentially stronger bokeh effects.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "disparity_map": ("IMAGE",),
                "focus_point": ("FOCUS_POINT",),
            },
            "optional": {
                "blur_strength": ("FLOAT", {
                    "default": 20.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.5,
                    "tooltip": "K value - controls bokeh intensity (0 = no blur)"
                }),
                "max_coc": ("FLOAT", {
                    "default": 100.0,
                    "min": 1.0,
                    "max": 500.0,
                    "step": 1.0,
                    "tooltip": "Maximum circle of confusion (blur radius)"
                }),
                "use_raw_disparity": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, expects raw disparity values. If False, expects normalized 0-1 depth."
                }),
                "normalize_for_genfocus": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, stretch defocus map to full 0-1 range for stronger bokeh. Original Genfocus uses False."
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("defocus_preview", "defocus_map", "defocus_raw")
    FUNCTION = "compute"
    CATEGORY = "Refocus/Defocus"
    DESCRIPTION = "Compute defocus map from disparity and focus point. Preview is normalized for visualization, map is for Genfocus, raw shows actual blur values."
    
    def compute(self, disparity_map, focus_point, blur_strength=20.0, max_coc=100.0, use_raw_disparity=False, normalize_for_genfocus=False):
        B, H, W, C = disparity_map.shape
        
        # Get disparity values (use first channel if grayscale or average if RGB)
        if C == 1:
            disp = disparity_map[:, :, :, 0].cpu().numpy()
        else:
            disp = disparity_map[:, :, :, 0].cpu().numpy()  # Use R channel
        
        focus_x = focus_point["x"]
        focus_y = focus_point["y"]
        
        # If normalized depth (0-1), convert to disparity-like values
        if not use_raw_disparity:
            # Assume higher values = closer (typical depth convention)
            # Invert to get disparity-like (closer = higher disparity)
            disp = disp  # Already in usable form for relative comparison
        
        # Get disparity at focus point
        disp_focus = disp[:, focus_y, focus_x]
        
        # Compute defocus map: K * |disp - disp_focus|
        # Broadcast focus disparity across spatial dimensions
        dmf = np.abs(blur_strength * (disp - disp_focus[:, np.newaxis, np.newaxis]))
        
        # Clamp to max CoC for the Genfocus input (0-1 range) - ORIGINAL behavior
        defocus_for_genfocus = np.clip(dmf / max_coc, 0.0, 1.0)
        
        # Optionally normalize to full 0-1 range for stronger effect
        if normalize_for_genfocus:
            dmf_min = defocus_for_genfocus.min()
            dmf_max = defocus_for_genfocus.max()
            if dmf_max > dmf_min:
                defocus_for_genfocus = (defocus_for_genfocus - dmf_min) / (dmf_max - dmf_min)
                print(f"[DefocusMap] NORMALIZED output: stretched {dmf_min:.3f}-{dmf_max:.3f} to 0-1")
        
        # Create NORMALIZED preview (stretch to full 0-1 range for visualization)
        dmf_min = dmf.min()
        dmf_max = dmf.max()
        if dmf_max > dmf_min:
            defocus_normalized_preview = (dmf - dmf_min) / (dmf_max - dmf_min)
        else:
            defocus_normalized_preview = np.zeros_like(dmf)
        
        # Create RAW visualization (shows actual blur values before clamping)
        raw_preview = np.clip(dmf / max_coc, 0.0, 1.0)  # Same as genfocus but for viz
        
        # Convert to ComfyUI image format (B,H,W,C)
        # Preview: normalized 0-1 for full contrast visualization
        preview_rgb = np.stack([defocus_normalized_preview, defocus_normalized_preview, defocus_normalized_preview], axis=-1)
        defocus_preview = torch.from_numpy(preview_rgb.astype(np.float32))
        
        # Defocus map for Genfocus: clamped to max_coc (optionally normalized)
        genfocus_rgb = np.stack([defocus_for_genfocus, defocus_for_genfocus, defocus_for_genfocus], axis=-1)
        defocus_map_tensor = torch.from_numpy(genfocus_rgb.astype(np.float32))
        
        # Raw: shows actual blur values (before any normalization)
        raw_rgb = np.stack([raw_preview, raw_preview, raw_preview], axis=-1)
        defocus_raw_tensor = torch.from_numpy(raw_rgb.astype(np.float32))
        
        print(f"[DefocusMap] Blur strength K={blur_strength}, Max CoC={max_coc}")
        print(f"[DefocusMap] Raw blur range: {dmf.min():.3f} - {dmf.max():.3f}")
        print(f"[DefocusMap] Genfocus input range: {defocus_for_genfocus.min():.3f} - {defocus_for_genfocus.max():.3f}")
        print(f"[DefocusMap] Focus disparity: {disp_focus[0]:.4f}, Disparity range: {disp.min():.4f} - {disp.max():.4f}")
        
        return (defocus_preview, defocus_map_tensor, defocus_raw_tensor)


class FocusPointFromMask:
    """
    Extract focus point from a mask input.
    
    Use this node with Interactive SAM Detector, MaskPainter, or any mask source
    to interactively select your focus point. The focus point is calculated as
    the centroid (weighted center) of the mask.
    
    Workflow:
    1. Connect image to this node
    2. Connect mask from SAM Detector or MaskPainter
    3. Click on your focus target to generate the mask
    4. The focus point is automatically extracted
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK", {
                    "tooltip": "Mask from SAM, MaskPainter, or any mask source"
                }),
            },
            "optional": {
                "fallback_x": ("FLOAT", {
                    "default": 50.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Fallback X% if mask is empty"
                }),
                "fallback_y": ("FLOAT", {
                    "default": 50.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Fallback Y% if mask is empty"
                }),
            }
        }
    
    RETURN_TYPES = ("FOCUS_POINT", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("focus_point", "preview", "x", "y")
    FUNCTION = "extract_focus"
    CATEGORY = "Refocus/Defocus"
    DESCRIPTION = "Extract focus point from mask centroid. Use with SAM Detector or MaskPainter for interactive selection."
    
    def _get_mask_centroid(self, mask):
        """Calculate the centroid of the mask (weighted center of mass)."""
        mask_np = mask.cpu().numpy()
        
        # Handle batch dimension
        if mask_np.ndim == 3:
            mask_np = mask_np[0]  # Use first mask in batch
        
        # Find non-zero pixels
        ys, xs = np.where(mask_np > 0.5)
        
        if len(xs) == 0:
            return None, None  # Empty mask
        
        # Calculate centroid (weighted by mask values for soft masks)
        weights = mask_np[ys, xs]
        total_weight = weights.sum()
        
        if total_weight == 0:
            return None, None
        
        centroid_x = int(np.round((xs * weights).sum() / total_weight))
        centroid_y = int(np.round((ys * weights).sum() / total_weight))
        
        return centroid_x, centroid_y
    
    def extract_focus(self, image, mask, fallback_x=50.0, fallback_y=50.0):
        B, H, W, C = image.shape
        
        # Get centroid from mask
        focus_x, focus_y = self._get_mask_centroid(mask)
        
        if focus_x is None:
            # Fallback to percentage
            focus_x = int(W * fallback_x / 100.0)
            focus_y = int(H * fallback_y / 100.0)
            print("[FocusFromMask] Warning: Mask is empty, using fallback position")
        else:
            print(f"[FocusFromMask] Extracted centroid: ({focus_x}, {focus_y})")
        
        # Clamp to image bounds
        focus_x = max(0, min(focus_x, W - 1))
        focus_y = max(0, min(focus_y, H - 1))
        
        focus_point = {
            "x": focus_x,
            "y": focus_y,
            "width": W,
            "height": H,
        }
        
        # Create preview with focus point and mask overlay
        preview = image.clone()
        
        # Overlay mask as semi-transparent green
        mask_np = mask.cpu().numpy()
        if mask_np.ndim == 3:
            mask_expanded = np.expand_dims(mask_np, -1)  # B,H,W,1
        else:
            mask_expanded = np.expand_dims(np.expand_dims(mask_np, 0), -1)  # 1,H,W,1
        
        mask_overlay = torch.from_numpy(mask_expanded.astype(np.float32))
        
        # Blend: show green tint where mask is active
        for i in range(B):
            if i < mask_overlay.shape[0]:
                m = mask_overlay[i, :, :, 0]
            else:
                m = mask_overlay[0, :, :, 0]
            
            # Green tint for mask area
            preview[i, :, :, 1] = preview[i, :, :, 1] * (1 - m * 0.3) + m * 0.3
        
        # Draw red crosshair at focus point
        r = 12  # Radius
        line_width = 3
        for i in range(B):
            # Horizontal line
            y_start = max(0, focus_y - line_width // 2)
            y_end = min(H, focus_y + line_width // 2 + 1)
            x_start = max(0, focus_x - r)
            x_end = min(W, focus_x + r + 1)
            preview[i, y_start:y_end, x_start:x_end, 0] = 1.0
            preview[i, y_start:y_end, x_start:x_end, 1] = 0.0
            preview[i, y_start:y_end, x_start:x_end, 2] = 0.0
            
            # Vertical line
            y_start = max(0, focus_y - r)
            y_end = min(H, focus_y + r + 1)
            x_start = max(0, focus_x - line_width // 2)
            x_end = min(W, focus_x + line_width // 2 + 1)
            preview[i, y_start:y_end, x_start:x_end, 0] = 1.0
            preview[i, y_start:y_end, x_start:x_end, 1] = 0.0
            preview[i, y_start:y_end, x_start:x_end, 2] = 0.0
        
        return (focus_point, preview, focus_x, focus_y)
