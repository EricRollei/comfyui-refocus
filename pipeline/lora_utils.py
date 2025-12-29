"""
LoRA utilities for Genfocus multi-adapter switching.
Ported from rayray9999/Genfocus
"""

import torch
from contextlib import contextmanager
from typing import List, Optional

try:
    from peft.tuners.tuners_utils import BaseTunerLayer
except ImportError:
    # Fallback if peft not installed - create a dummy class
    class BaseTunerLayer:
        pass


@contextmanager
def specify_lora(lora_modules: tuple, specified_lora: Optional[str]):
    """
    Context manager to temporarily set which LoRA adapter is active by adjusting scaling.
    
    This allows different branches of the transformer to use different LoRA adapters
    during the same forward pass. Used for multi-conditional generation where each
    condition branch may need its own adapter.
    
    The implementation sets scaling[adapter] = 1 for the specified adapter and 0 for others.
    When specified_lora is None, ALL adapters are set to 0 (no LoRA applied).
    
    Args:
        lora_modules: Tuple of modules that may have LoRA adapters
        specified_lora: Name of the adapter to activate (e.g., "deblurring", "bokeh")
                       If None, ALL adapters are disabled (scaling = 0).
    
    Example:
        with specify_lora((attn.to_q, attn.to_k, attn.to_v), "bokeh"):
            query = attn.to_q(hidden_state)
            key = attn.to_k(hidden_state)
            value = attn.to_v(hidden_state)
    """
    # NOTE: When specified_lora is None, we still need to set all scaling to 0
    # to disable LoRA for the main branch. This matches the original Genfocus behavior.
    
    # Debug: Show scaling info on first call for each adapter type
    debug_key = f'_debug_{specified_lora}'
    if not hasattr(specify_lora, debug_key):
        setattr(specify_lora, debug_key, True)
        target_scaling = '0.0 for ALL' if specified_lora is None else f'1.0 for {specified_lora}, 0.0 for others'
        print(f"[specify_lora] adapter='{specified_lora}' → scaling: {target_scaling}")
    
    # Filter to only valid lora modules
    valid_lora_modules = [m for m in lora_modules if isinstance(m, BaseTunerLayer)]
    
    # Debug: One-time check that modules are found
    if valid_lora_modules and not hasattr(specify_lora, '_modules_confirmed'):
        specify_lora._modules_confirmed = True
        m = valid_lora_modules[0]
        print(f"[specify_lora] Found {len(valid_lora_modules)} LoRA modules, active_adapters={list(m.active_adapters)}, scaling keys={list(m.scaling.keys())}")
    
    if not valid_lora_modules:
        # No LoRA modules found - this is a problem if we expected adapters
        # Add a one-time warning
        if not hasattr(specify_lora, '_warned_no_lora'):
            print(f"[specify_lora] Warning: No BaseTunerLayer modules found for adapter '{specified_lora}'")
            print(f"[specify_lora] Modules checked: {[type(m).__name__ for m in lora_modules]}")
            specify_lora._warned_no_lora = True
        yield
        return
    
    # Save original scales
    original_scales = []
    for module in valid_lora_modules:
        if hasattr(module, 'scaling') and hasattr(module, 'active_adapters'):
            scales = {
                adapter: module.scaling[adapter]
                for adapter in module.active_adapters
                if adapter in module.scaling
            }
            original_scales.append(scales)
        else:
            original_scales.append({})
    
    try:
        # Enter context: adjust scaling - set specified adapter to 1, others to 0
        for module in valid_lora_modules:
            if hasattr(module, 'scaling') and hasattr(module, 'active_adapters'):
                for adapter in module.active_adapters:
                    if adapter in module.scaling:
                        module.scaling[adapter] = 1.0 if adapter == specified_lora else 0.0
        yield
    finally:
        # Exit context: restore original scales
        for module, scales in zip(valid_lora_modules, original_scales):
            if hasattr(module, 'scaling'):
                for adapter, scale in scales.items():
                    if adapter in module.scaling:
                        module.scaling[adapter] = scale


def load_genfocus_lora(pipeline, lora_path: str, adapter_name: str, weight: float = 1.0):
    """
    Load a Genfocus LoRA adapter into the pipeline.
    
    Args:
        pipeline: FluxPipeline instance
        lora_path: Path to the LoRA safetensors file
        adapter_name: Name to give this adapter (e.g., "deblurring", "bokeh")
        weight: LoRA weight multiplier
    
    Returns:
        True if successful, False otherwise
    """
    try:
        pipeline.load_lora_weights(
            lora_path,
            adapter_name=adapter_name,
        )
        pipeline.set_adapters([adapter_name], adapter_weights=[weight])
        print(f"[Genfocus] Loaded LoRA adapter '{adapter_name}' from {lora_path}")
        return True
    except Exception as e:
        print(f"[Genfocus] Failed to load LoRA '{adapter_name}': {e}")
        return False


def switch_adapter(pipeline, adapter_name: str):
    """
    Switch the active LoRA adapter.
    
    Args:
        pipeline: FluxPipeline instance
        adapter_name: Name of the adapter to activate
    """
    try:
        pipeline.set_adapters([adapter_name])
        print(f"[Genfocus] Switched to adapter: {adapter_name}")
    except Exception as e:
        print(f"[Genfocus] Failed to switch adapter: {e}")
