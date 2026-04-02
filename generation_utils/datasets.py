"""Dataset utilities for conditional diffusion training and sampling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


class SpeckleAugmentedDataset(Dataset):
    """Load propagated intensity fields and downsample them to a learning grid."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        root_dir: Path | str,
        transform=None,
        downsample_size: tuple[int, int] = (256, 256),
        final_size: tuple[int, int] = (256, 256),
        shift: int = 0,
        shift_mode: str = "fixed",
        is_train: bool = True,
        samples_per_class: int | None = None,
    ) -> None:
        if samples_per_class is not None and samples_per_class > 0:
            label_col = metadata.columns[1]
            metadata = (
                metadata.groupby(label_col, group_keys=False)
                .head(samples_per_class)
                .reset_index(drop=True)
            )

        self.metadata = metadata.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_train = is_train
        self.shift_mode = str(shift_mode)
        self.down_h, self.down_w = downsample_size
        self.final_h, self.final_w = final_size
        self.shift = int(shift)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int):
        img_path = self.root_dir / self.metadata.iloc[idx, 0]
        label = int(self.metadata.iloc[idx, 1]) - 1
        full_image = np.load(img_path)
        full_h, full_w = full_image.shape

        image_tensor = torch.from_numpy(full_image.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        kernel_h = full_h // self.down_h
        kernel_w = full_w // self.down_w
        downsampler = torch.nn.AvgPool2d(kernel_size=(kernel_h, kernel_w))
        image_tensor = downsampler(image_tensor).squeeze(0)

        _, h, w = image_tensor.shape
        start_y = (h - self.final_h) // 2
        start_x = (w - self.final_w) // 2

        if self.is_train and self.shift != 0:
            if self.shift_mode == "fixed":
                shift_x = self.shift
                shift_y = self.shift
            elif self.shift_mode == "random":
                shift_x = torch.randint(-abs(self.shift), abs(self.shift) + 1, (1,)).item()
                shift_y = torch.randint(-abs(self.shift), abs(self.shift) + 1, (1,)).item()
            else:
                raise ValueError(f"Unknown shift_mode: {self.shift_mode}")

            start_x = max(0, min(start_x + shift_x, w - self.final_w))
            start_y = max(0, min(start_y + shift_y, h - self.final_h))

        final_image = image_tensor[:, start_y : start_y + self.final_h, start_x : start_x + self.final_w]
        if self.transform is not None:
            final_image = self.transform(final_image)
        return final_image, torch.tensor(label, dtype=torch.long)


class SpeckleDiffusionDataset(Dataset):
    """Load intensity fields, downsample them, and normalize them to [-1, 1]."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        root_dir: Path | str,
        stats_path: Path | str,
        final_size: tuple[int, int] = (256, 256),
        shift: int = 0,
        is_train: bool = True,
    ) -> None:
        self.metadata = metadata_df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.final_h, self.final_w = final_size
        self.is_train = is_train
        self.shift_amount = int(shift)

        with open(stats_path, "r", encoding="utf-8") as handle:
            stats = json.load(handle)
        self.min_val = float(stats["intensity_min"])
        self.max_val = float(stats["intensity_max"])

        img_path = self.root_dir / self.metadata.iloc[0, 0]
        image = np.load(img_path)
        self.full_h, self.full_w = image.shape
        if self.full_h % self.final_h != 0:
            raise ValueError("The full image height must be divisible by the target height.")
        self.downsampler = torch.nn.AvgPool2d(kernel_size=self.full_h // self.final_h)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int):
        img_path = self.root_dir / self.metadata.iloc[idx, 0]
        label = int(self.metadata.iloc[idx, 1]) - 1

        image = np.load(img_path)
        image_tensor = torch.from_numpy(image.astype(np.float32))

        if self.is_train and self.shift_amount != 0:
            image_tensor = torch.roll(
                image_tensor,
                shifts=(self.shift_amount, self.shift_amount),
                dims=(0, 1),
            )

        downsampled = self.downsampler(image_tensor.unsqueeze(0).unsqueeze(0))
        normalized = 2.0 * (downsampled - self.min_val) / (self.max_val - self.min_val) - 1.0
        final_image = normalized.squeeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return {"target": final_image, "label": label_tensor}


def calculate_intensity_range(
    metadata_df: pd.DataFrame,
    root_dir: Path | str,
    final_size: tuple[int, int],
    shift: int = 0,
    batch_size: int = 64,
    is_train: bool = False,
    cache_filename: str = "dataset_intensity_stats.json",
    force_recalc: bool = False,
) -> tuple[float, float]:
    """Compute and cache the intensity range after preprocessing."""

    root_dir = Path(root_dir)
    cache_path = root_dir / cache_filename
    if cache_path.exists() and not force_recalc:
        with open(cache_path, "r", encoding="utf-8") as handle:
            stats = json.load(handle)
        return float(stats["intensity_min"]), float(stats["intensity_max"])

    dataset = SpeckleAugmentedDataset(
        metadata=metadata_df,
        root_dir=root_dir,
        final_size=final_size,
        shift=shift,
        is_train=is_train,
        transform=None,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    intensity_min = float("inf")
    intensity_max = float("-inf")
    for images, _ in tqdm(loader, desc="Scanning dataset", leave=False):
        intensity_min = min(intensity_min, float(torch.min(images).item()))
        intensity_max = max(intensity_max, float(torch.max(images).item()))

    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "intensity_min": float(intensity_min),
                "intensity_max": float(intensity_max),
                "final_size": list(final_size),
                "shift": int(shift),
            },
            handle,
            indent=2,
        )
    return float(intensity_min), float(intensity_max)
