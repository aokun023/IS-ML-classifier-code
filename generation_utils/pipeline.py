"""Training and sampling pipeline for conditional diffusion generation."""

from __future__ import annotations

import gc
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from diffusers import FlowMatchEulerDiscreteScheduler, UNet2DModel
from ema_pytorch import EMA
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .datasets import SpeckleAugmentedDataset, calculate_intensity_range
from .diffusion import build_noise_scheduler, build_unet, get_diffusion_variables, load_compiled_state_dict
from .losses import BregmanCharbonnierLoss, FrequencyBregmanPureL1Loss, LinearMinMaxNormalize


def _extract_dataset_parameter(dataset_name: str, key: str) -> str:
    """Extract one parameter value from a dataset directory name."""

    match = re.search(rf"{re.escape(key)}-([^_]+)", dataset_name)
    return match.group(1) if match is not None else "unknown"


@dataclass
class PipelineConfig:
    """Configuration for conditional diffusion training and sample generation."""

    repo_root: Path
    dataset_path: Path
    metadata_filename: str
    dataset_tag: str | None
    input_resolution_tag: str
    model_type: str
    prediction_type: str
    loss_target_type: str
    pixel_loss_type: str
    lambda_freq: float
    image_size: int
    num_class_embed: int
    random_seed: int
    val_split_ratio: float
    samples_per_class: int
    training_samples_per_class: int
    validation_samples_per_class: int
    train_batch_size: int
    val_batch_size: int
    pretrain_epochs: int
    pretrain_lr: float
    pretrain_lr_patience: int
    pretrain_lr_factor: float
    num_samples_to_generate: int
    generation_batch_size: int
    append_generated_samples: bool
    class_ids: list[int]
    model_config: dict
    p_uncond: float
    guidance_strength: float

    def __post_init__(self) -> None:
        self.metadata_csv = self.dataset_path / self.metadata_filename
        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_csv}")
        base_dataset_tag = self.dataset_tag if self.dataset_tag is not None else self.dataset_path.name
        self.dataset_name_tag = f"{base_dataset_tag}_InputRes-{self.input_resolution_tag}"
        metadata_df = pd.read_csv(self.metadata_csv)
        self.intensity_min, self.intensity_max = calculate_intensity_range(
            metadata_df,
            self.dataset_path,
            final_size=(self.image_size, self.image_size),
            shift=0,
            is_train=False,
        )
        self.normalize_transform = LinearMinMaxNormalize(self.intensity_min, self.intensity_max)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def output_root(self) -> Path:
        dataset_name = self.dataset_path.name
        z_value = _extract_dataset_parameter(dataset_name, "z")
        sigma_value = _extract_dataset_parameter(dataset_name, "sigma")

        base = Path(
          "generation_"
          f"z-{z_value}_"
          f"sigma-{sigma_value}_"
          f"pred-{self.prediction_type}_"
          f"loss-{self.loss_target_type}_"
          f"lambda-{self.lambda_freq}_"
          f"res-{self.image_size}_"
          f"classes-{self.num_class_embed}_"
          f"puncond-{self.p_uncond}_"
          f"cfgw-{self.guidance_strength}_"
          f"seed-{self.random_seed}"
        )
        return self.repo_root / "results" / base


def pretrain_generator(config: PipelineConfig, accelerator: Accelerator) -> Path:
    """Train the conditional generator and return the best checkpoint path."""
    output_dir = config.output_root / "stage1_pretrain"
    best_model_path = output_dir / "best_model.bin"
    if best_model_path.exists():
        return best_model_path

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    full_metadata = pd.read_csv(config.metadata_csv)
    train_meta, val_meta = train_test_split(
        full_metadata,
        test_size=config.val_split_ratio,
        random_state=config.random_seed,
    )

    train_dataset = SpeckleAugmentedDataset(
        metadata=train_meta,
        root_dir=config.dataset_path,
        final_size=(config.image_size, config.image_size),
        transform=config.normalize_transform,
        samples_per_class=config.training_samples_per_class,
    )
    val_dataset = SpeckleAugmentedDataset(
        metadata=val_meta,
        root_dir=config.dataset_path,
        final_size=(config.image_size, config.image_size),
        transform=config.normalize_transform,
        samples_per_class=config.validation_samples_per_class,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False, num_workers=0)

    model = build_unet(config.model_config, config.num_class_embed)
    noise_scheduler = build_noise_scheduler(config.model_type, config.prediction_type)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.pretrain_lr)
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.pretrain_lr_factor,
        patience=config.pretrain_lr_patience,
    )
    ema_model = EMA(model, beta=0.995, update_every=10).to(accelerator.device)

    if config.pixel_loss_type == "MSE":
        pixel_loss_fn = nn.MSELoss()
    elif config.pixel_loss_type == "CharbonnierBregman":
        pixel_loss_fn = BregmanCharbonnierLoss().to(accelerator.device)
    else:
        raise ValueError(f"Unknown pixel loss: {config.pixel_loss_type}")
    freq_loss_fn = FrequencyBregmanPureL1Loss().to(accelerator.device)

    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    best_val_loss = float("inf")

    for _ in range(config.pretrain_epochs):
        model.train()
        for clean_images, labels in tqdm(train_loader, disable=not accelerator.is_local_main_process, leave=False):
          
            labels = labels.to(accelerator.device)
            labels_cfg = labels.clone()
            if config.guidance_strength != 0:
              null_id = config.num_class_embed - 1
              drop_mask = torch.rand(labels.shape, device=labels.device) < config.p_uncond
              labels_cfg[drop_mask] = null_id

            with accelerator.accumulate(model):
                noise = torch.randn_like(clean_images)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (clean_images.shape[0],),
                    device=clean_images.device,
                ).long()

                if config.model_type == "FlowMatching":
                    t_norm = timesteps.float().view(-1, 1, 1, 1) / noise_scheduler.config.num_train_timesteps
                    noisy_images = (1 - t_norm) * clean_images + t_norm * noise
                else:
                    noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                model_output = model(noisy_images, timesteps, class_labels=labels_cfg).sample
                preds, targets = get_diffusion_variables(
                    model_output,
                    noisy_images,
                    clean_images,
                    noise,
                    timesteps,
                    noise_scheduler,
                    config.model_type,
                    config.prediction_type,
                )
                pred_main = preds[config.loss_target_type]
                target_main = targets[config.loss_target_type]
                main_loss = pixel_loss_fn(pred_main, target_main)
                freq_loss = freq_loss_fn(pred_main, target_main)
                loss = main_loss + config.lambda_freq * freq_loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients and accelerator.is_main_process:
                ema_model.update()

        model.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for clean_images, labels in tqdm(val_loader, disable=not accelerator.is_local_main_process, leave=False):
                labels = labels.to(accelerator.device)
                noise = torch.randn_like(clean_images)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (clean_images.shape[0],),
                    device=clean_images.device,
                ).long()
                if config.model_type == "FlowMatching":
                    t_norm = timesteps.float().view(-1, 1, 1, 1) / noise_scheduler.config.num_train_timesteps
                    noisy_images = (1 - t_norm) * clean_images + t_norm * noise
                else:
                    noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                model_output = model(noisy_images, timesteps, class_labels=labels).sample
                preds, targets = get_diffusion_variables(
                    model_output,
                    noisy_images,
                    clean_images,
                    noise,
                    timesteps,
                    noise_scheduler,
                    config.model_type,
                    config.prediction_type,
                )
                pred_main = preds[config.loss_target_type]
                target_main = targets[config.loss_target_type]
                val_loss = pixel_loss_fn(pred_main, target_main) + config.lambda_freq * freq_loss_fn(pred_main, target_main)
                val_loss_epoch += accelerator.gather(val_loss).mean().item()

        avg_val_loss = val_loss_epoch / max(len(val_loader), 1)
        if accelerator.is_main_process:
            lr_scheduler.step(avg_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                unwrapped_model = accelerator.unwrap_model(ema_model.ema_model)
                torch.save(unwrapped_model.state_dict(), best_model_path)

    if accelerator.is_main_process and not best_model_path.exists():
        unwrapped_model = accelerator.unwrap_model(ema_model.ema_model)
        torch.save(unwrapped_model.state_dict(), best_model_path)

    del model, optimizer, ema_model, train_loader, val_loader
    accelerator.clear()
    gc.collect()
    torch.cuda.empty_cache()
    return best_model_path


def generate_samples_for_class(class_label: int, config: PipelineConfig, model_path: Path) -> Path:
    """Generate stage-5 synthetic samples for one class."""
    output_dir = config.output_root / f"class-{class_label}" / "stage5_pretrained_data"
    metadata_path = output_dir / "metadata.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_records = []
    existing_count = 0
    if metadata_path.exists() and config.append_generated_samples:
        existing_df = pd.read_csv(metadata_path)
        existing_records = existing_df.to_dict("records")
        existing_count = len(existing_records)
    elif metadata_path.exists() and not config.append_generated_samples:
        return output_dir

    model = UNet2DModel(
        sample_size=config.model_config["sample_size"],
        in_channels=config.model_config["in_channels"],
        out_channels=config.model_config["out_channels"],
        layers_per_block=config.model_config["layers_per_block"],
        block_out_channels=config.model_config["block_out_channels"],
        down_block_types=config.model_config["down_block_types"],
        up_block_types=config.model_config["up_block_types"],
        num_class_embeds=config.num_class_embed,
    )
    state_dict = torch.load(model_path, map_location="cpu")
    load_compiled_state_dict(model, state_dict)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model.to(config.device, dtype=dtype)
    model.eval()

    noise_scheduler = build_noise_scheduler(config.model_type, config.prediction_type)
    noise_scheduler.set_timesteps(1000)
    generator = torch.Generator(device=config.device).manual_seed(config.random_seed)

    metadata_records = list(existing_records)
    sample_counter = existing_count
    target_total = existing_count + config.num_samples_to_generate

    while sample_counter < target_total:
        batch_size = min(config.generation_batch_size, target_total - sample_counter)
        latents = torch.randn(
            (batch_size, 1, config.image_size, config.image_size),
            generator=generator,
            device=config.device,
            dtype=dtype,
        )
        labels = torch.full((batch_size,), class_label - 1, dtype=torch.long, device=config.device)

        for t in noise_scheduler.timesteps:
            latent_input = noise_scheduler.scale_model_input(latents, t) if hasattr(noise_scheduler, "scale_model_input") else latents
            t_tensor = t if torch.is_tensor(t) else torch.tensor([t], device=config.device)
            with torch.no_grad():
                if config.guidance_strength == 0:
                  model_output = model(latent_input, t_tensor, class_labels=labels).sample
                else:
                  null_id = config.num_class_embed - 1
                  labels_u = torch.full_like(labels, null_id)

                  model_output_c = model(latent_input, t_tensor, class_labels=labels).sample
                  model_output_u = model(latent_input, t_tensor, class_labels=labels_u).sample

                  model_output = (1.0 + config.guidance_strength) * model_output_c - config.guidance_strength * model_output_u
                  
                if isinstance(noise_scheduler, FlowMatchEulerDiscreteScheduler) and config.prediction_type == "sample":
                    sigma = t_tensor / noise_scheduler.config.num_train_timesteps
                    model_output = (latents - model_output) / (sigma + 1e-5)
                try:
                    latents = noise_scheduler.step(model_output, t, latents).prev_sample
                except IndexError:
                    break

        images = latents.to(torch.float32)
        images = images * (config.intensity_max - config.intensity_min) / 2 + (config.intensity_max + config.intensity_min) / 2
        images = images.clamp(min=config.intensity_min, max=config.intensity_max).cpu().numpy().squeeze(axis=1)

        for image in images:
            filename = f"generated_class-{class_label}_sample-{sample_counter:04d}.npy"
            np.save(output_dir / filename, image)
            metadata_records.append({"filepath": filename, "class_label": class_label})
            sample_counter += 1

    pd.DataFrame(metadata_records, columns=["filepath", "class_label"]).to_csv(metadata_path, index=False)
    del model
    torch.cuda.empty_cache()
    return output_dir


def run_generation_pipeline(config: PipelineConfig, accelerator: Accelerator) -> dict:
    """Run pretraining and per-class sample generation."""
    start_time = time.time()
    best_model_path = pretrain_generator(config, accelerator)
    summary = {"best_model_path": str(best_model_path), "generated_class_dirs": {}}

    for class_label in config.class_ids:
        class_dir = generate_samples_for_class(class_label, config, best_model_path)
        summary["generated_class_dirs"][str(class_label)] = str(class_dir)

    summary["elapsed_seconds"] = float(time.time() - start_time)
    summary_path = config.output_root / "generation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
