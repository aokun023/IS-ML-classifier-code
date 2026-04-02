"""Model and scheduler utilities for conditional diffusion experiments."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from diffusers import DDPMScheduler, FlowMatchEulerDiscreteScheduler, UNet2DModel


def configure_model_memory(model: UNet2DModel) -> UNet2DModel:
    """Enable optional memory-saving features when available."""
    if hasattr(model, "enable_xformers_memory_efficient_attention"):
        try:
            import xformers  # noqa: F401

            model.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    if hasattr(model, "enable_gradient_checkpointing"):
        try:
            model.enable_gradient_checkpointing()
        except Exception:
            pass
    return model


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_compiled_state_dict(model: nn.Module, state_dict: dict) -> nn.Module:
    """Load a state dict produced with or without `torch.compile`."""
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            cleaned_state_dict[key.replace("_orig_mod.", "")] = value
        else:
            cleaned_state_dict[key] = value
    model.load_state_dict(cleaned_state_dict)
    return model


def build_unet(model_config: dict, num_class_embed: int) -> UNet2DModel:
    """Construct the class-conditional U-Net used by the generator."""
    model = UNet2DModel(
        sample_size=model_config["sample_size"],
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        layers_per_block=model_config["layers_per_block"],
        block_out_channels=model_config["block_out_channels"],
        down_block_types=model_config["down_block_types"],
        up_block_types=model_config["up_block_types"],
        num_class_embeds=num_class_embed,
    )
    configure_model_memory(model)
    try:
        model = torch.compile(model, mode="reduce-overhead")
    except Exception:
        pass
    return model


def build_noise_scheduler(model_type: str, prediction_type: str):
    """Construct the scheduler corresponding to the chosen model type."""
    if model_type == "Diffusion":
        return DDPMScheduler(num_train_timesteps=1000, prediction_type=prediction_type)
    if model_type == "FlowMatching":
        return FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    raise ValueError(f"Unknown model_type: {model_type}")


def get_diffusion_variables(
    model_output: torch.Tensor,
    noisy_images: torch.Tensor,
    clean_images: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler,
    model_type: str,
    prediction_type: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Decode the model output into x0/epsilon/v variables."""
    if model_type == "FlowMatching":
        t_norm = timesteps.float() / noise_scheduler.config.num_train_timesteps
        t_norm = t_norm.view(-1, 1, 1, 1)
        alpha_t = 1.0 - t_norm
        sigma_t = t_norm
        target_x0 = clean_images
        target_eps = noise
        target_v = noise - clean_images

        if prediction_type == "v_prediction":
            pred_v = model_output
            pred_x0 = noisy_images - sigma_t * pred_v
            pred_eps = pred_v + pred_x0
        elif prediction_type == "sample":
            pred_x0 = model_output
            pred_v = (noisy_images - alpha_t * pred_x0) / (sigma_t + 1e-8)
            pred_eps = pred_v + pred_x0
        else:
            raise ValueError("FlowMatching supports only 'v_prediction' or 'sample'.")
    else:
        alphas_cumprod = noise_scheduler.alphas_cumprod.to(clean_images.device)
        alpha_t = alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1)
        sigma_t = (1 - alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)

        target_x0 = clean_images
        target_eps = noise
        target_v = alpha_t * noise - sigma_t * clean_images

        if prediction_type == "v_prediction":
            pred_v = model_output
            pred_x0 = alpha_t * noisy_images - sigma_t * pred_v
            pred_eps = sigma_t * noisy_images + alpha_t * pred_v
        elif prediction_type == "sample":
            pred_x0 = model_output
            pred_eps = (noisy_images - alpha_t * pred_x0) / sigma_t.clamp(min=0.05)
            pred_v = alpha_t * pred_eps - sigma_t * pred_x0
        elif prediction_type == "epsilon":
            pred_eps = model_output
            pred_x0 = (noisy_images - sigma_t * pred_eps) / alpha_t
            pred_v = alpha_t * pred_eps - sigma_t * pred_x0
        else:
            raise ValueError(f"Unknown prediction_type: {prediction_type}")

    preds = {"sample": pred_x0, "epsilon": pred_eps, "v_prediction": pred_v}
    targets = {"sample": target_x0, "epsilon": target_eps, "v_prediction": target_v}
    return preds, targets
