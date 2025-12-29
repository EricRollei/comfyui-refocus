"""
Generate function for Genfocus multi-conditional generation.
Ported from rayray9999/Genfocus

This is the main entry point for running the Genfocus pipeline.
"""

import torch
import numpy as np
from typing import List, Union, Optional, Dict, Any, Callable, Tuple

try:
    from diffusers.pipelines import FluxPipeline
    from diffusers.pipelines.flux.pipeline_flux import (
        FluxPipelineOutput,
        calculate_shift,
        retrieve_timesteps,
    )
    from diffusers.models.attention_processor import Attention
except ImportError:
    FluxPipeline = None
    FluxPipelineOutput = None
    calculate_shift = None
    retrieve_timesteps = None
    Attention = None

from .condition import Condition
from .transformer_forward import transformer_forward


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def generate(
    pipeline: "FluxPipeline",
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    num_inference_steps: int = 28,
    timesteps: Optional[List[int]] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = None,
    max_sequence_length: int = 512,
    # Genfocus-specific parameters
    main_adapter: Optional[str] = None,
    conditions: Optional[List[Condition]] = None,
    image_guidance_scale: float = 1.0,
    transformer_kwargs: Optional[Dict[str, Any]] = None,
    kv_cache: bool = False,
    latent_mask: Optional[torch.Tensor] = None,
    tile_size: int = 32,
    **params: dict,
) -> "FluxPipelineOutput":
    """
    Generate images using the Genfocus multi-conditional pipeline.
    
    This is a custom generate function that extends FluxPipeline to support:
    1. Multiple image conditions (e.g., blurry input + defocus map)
    2. Per-condition LoRA adapters
    3. KV caching for efficiency
    
    Args:
        pipeline: FluxPipeline instance with loaded LoRAs
        prompt: Text prompt for generation
        height, width: Output image dimensions
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        conditions: List of Condition objects for multi-conditional generation
        main_adapter: LoRA adapter for main generation branch
        image_guidance_scale: Scale for image conditioning (1.0 = no uncond)
        kv_cache: Enable KV caching for condition branches
        
    Returns:
        FluxPipelineOutput with generated images
    """
    if callback_on_step_end_tensor_inputs is None:
        callback_on_step_end_tensor_inputs = ["latents"]
    if conditions is None:
        conditions = []
    if transformer_kwargs is None:
        transformer_kwargs = {}
    
    self = pipeline
    
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor
    
    # Validate inputs
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )
    
    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    
    # Determine batch size
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    
    device = self._execution_device
    
    # Encode prompt
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )
    
    # Prepare latents
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, latent_image_ids = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    
    print(f"[Genfocus Generate] Main latents: shape={latents.shape}, ids shape={latent_image_ids.shape}")
    print(f"[Genfocus Generate] Target size: {width}x{height}, main_adapter={main_adapter}")
    
    # Apply latent mask if provided
    if latent_mask is not None:
        latent_mask_flat = latent_mask.T.reshape(-1)
        latents = latents[:, latent_mask_flat]
        latent_image_ids = latent_image_ids[latent_mask_flat]
    
    # Encode conditions
    c_latents, uc_latents, c_ids = [], [], []
    c_timesteps, c_projections, c_guidances, c_adapters = [], [], [], []
    complement_cond = None
    
    print(f"[Genfocus Generate] Encoding {len(conditions)} conditions...")
    for idx, condition in enumerate(conditions):
        tokens, ids = condition.encode(self)
        print(f"  Condition {idx}: tokens shape={tokens.shape}, ids shape={ids.shape}, adapter={condition.adapter}")
        c_latents.append(tokens)
        
        if image_guidance_scale != 1.0:
            uc_latents.append(condition.encode(self, empty=True)[0])
        
        c_ids.append(ids)
        c_timesteps.append(torch.zeros([1], device=device))
        c_projections.append(pooled_prompt_embeds)
        c_guidances.append(torch.ones([1], device=device))
        c_adapters.append(condition.adapter)
        
        if condition.is_complement:
            complement_cond = (tokens, ids)
    
    # Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )
    timesteps_tensor, num_inference_steps = retrieve_timesteps(
        self.scheduler, num_inference_steps, device, timesteps, sigmas, mu=mu
    )
    num_warmup_steps = max(
        len(timesteps_tensor) - num_inference_steps * self.scheduler.order, 0
    )
    self._num_timesteps = len(timesteps_tensor)
    
    # Setup KV cache if enabled
    if kv_cache:
        attn_counter = 0
        for module in self.transformer.modules():
            if isinstance(module, Attention):
                setattr(module, "cache_idx", attn_counter)
                attn_counter += 1
        kv_cond = [[[], []] for _ in range(attn_counter)]
        kv_uncond = [[[], []] for _ in range(attn_counter)]
        
        def clear_cache():
            for storage in [kv_cond, kv_uncond]:
                for keys, values in storage:
                    keys.clear()
                    values.clear()
    
    # Create group mask for cross-attention between branches
    branch_n = len(conditions) + 2  # main + text + conditions
    group_mask = torch.ones([branch_n, branch_n], dtype=torch.bool)
    # Disable cross-attention between different condition branches
    if len(conditions) > 0:
        group_mask[2:, 2:] = torch.diag(torch.tensor([1] * len(conditions)))
    
    # Denoising loop
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps_tensor):
            timestep = t.expand(latents.shape[0]).to(latents.dtype) / 1000
            
            # Handle guidance
            if self.transformer.config.guidance_embeds:
                guidance = torch.tensor([guidance_scale], device=device)
                guidance = guidance.expand(latents.shape[0])
            else:
                guidance = None
                c_guidances = [None for _ in c_guidances]
            
            if kv_cache:
                mode = "write" if i == 0 else "read"
                if mode == "write":
                    clear_cache()
            else:
                mode = None
            
            use_cond = not kv_cache or mode == "write"
            
            # Run transformer forward
            noise_pred = transformer_forward(
                self.transformer,
                image_features=[latents] + (c_latents if use_cond else []),
                text_features=[prompt_embeds],
                img_ids=[latent_image_ids] + (c_ids if use_cond else []),
                txt_ids=[text_ids],
                timesteps=[timestep, timestep] + (c_timesteps if use_cond else []),
                pooled_projections=[pooled_prompt_embeds] * 2 + (c_projections if use_cond else []),
                guidances=[guidance] * 2 + (c_guidances if use_cond else []),
                return_dict=False,
                adapters=[main_adapter] * 2 + (c_adapters if use_cond else []),
                cache_mode=mode if kv_cache else None,
                cache_storage=kv_cond if kv_cache else None,
                to_cache=[False, False] + [True] * len(c_latents) if use_cond else None,
                group_mask=group_mask,
                **transformer_kwargs,
            )[0]
            
            # Apply image guidance if configured
            if image_guidance_scale != 1.0:
                unc_pred = transformer_forward(
                    self.transformer,
                    image_features=[latents] + (uc_latents if use_cond else []),
                    text_features=[prompt_embeds],
                    img_ids=[latent_image_ids] + (c_ids if use_cond else []),
                    txt_ids=[text_ids],
                    timesteps=[timestep, timestep] + (c_timesteps if use_cond else []),
                    pooled_projections=[pooled_prompt_embeds] * 2 + (c_projections if use_cond else []),
                    guidances=[guidance] * 2 + (c_guidances if use_cond else []),
                    return_dict=False,
                    adapters=[main_adapter] * 2 + (c_adapters if use_cond else []),
                    cache_mode=mode if kv_cache else None,
                    cache_storage=kv_uncond if kv_cache else None,
                    to_cache=[False, False] + [True] * len(uc_latents) if use_cond else None,
                    **transformer_kwargs,
                )[0]
                
                noise_pred = unc_pred + image_guidance_scale * (noise_pred - unc_pred)
            
            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            
            # Callback
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)
            
            progress_bar.update()
    
    # Handle complement condition (for OminiControl2 style)
    if latent_mask is not None and complement_cond is not None:
        comp_latent, comp_ids = complement_cond
        all_ids = torch.cat([latent_image_ids, comp_ids], dim=0)
        shape = (all_ids.max(dim=0).values + 1).to(torch.long)
        H, W = shape[1].item(), shape[2].item()
        B, _, C = latents.shape
        
        canvas = latents.new_zeros(B, H * W, C)
        
        def _stash(canvas, tokens, ids, H, W):
            B, T, C = tokens.shape
            ids = ids.to(torch.long)
            flat_idx = (ids[:, 1] * W + ids[:, 2]).to(torch.long)
            canvas.view(B, -1, C).index_copy_(1, flat_idx, tokens)
        
        _stash(canvas, latents, latent_image_ids, H, W)
        _stash(canvas, comp_latent, comp_ids, H, W)
        latents = canvas.view(B, H * W, C)
    
    # Decode latents
    if output_type == "latent":
        image = latents
    else:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)
    
    self.maybe_free_model_hooks()
    
    if not return_dict:
        return (image,)
    
    return FluxPipelineOutput(images=image)
