"""Dataset and preprocessing utilities for IS-ML classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm


def compute_acf(image: np.ndarray | torch.Tensor) -> np.ndarray:
    """Compute the normalized periodic autocorrelation of one image."""

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image, dtype=np.float64)
    image = image - np.mean(image)
    spectrum = np.fft.fft2(image)
    acf = np.fft.ifft2(np.abs(spectrum) ** 2)
    acf = np.fft.fftshift(np.real(acf))
    center = acf[acf.shape[0] // 2, acf.shape[1] // 2]
    if abs(center) < 1e-12:
        return np.zeros_like(acf, dtype=np.float32)
    return (acf / center).astype(np.float32, copy=False)


class ACFTransform:
    """Replace the input intensity field by its normalized ACF."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        image = tensor.squeeze(0)
        acf = compute_acf(image)
        return torch.from_numpy(acf).unsqueeze(0)


class NormalizeTensor:
    """Channelwise affine normalization for one-channel inputs."""

    def __init__(self, mean: float, std: float, eps: float = 1e-8) -> None:
        self.mean = float(mean)
        self.std = float(std)
        self.eps = float(eps)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / max(self.std, self.eps)


class BaseSpeckleDataset(Dataset):
    """Shared crop-and-shift logic on a canvas image."""

    def __init__(
        self,
        transform=None,
        canvas_size: tuple[int, int] = (256, 256),
        final_size: tuple[int, int] = (64, 64),
        shift: int = 0,
        shift_mode: str = "fixed",
        is_train: bool = True,
        apply_shift_in_eval: bool = False,
    ) -> None:
        self.transform = transform
        self.canvas_h, self.canvas_w = canvas_size
        self.final_h, self.final_w = final_size
        self.shift = int(shift)
        self.shift_mode = str(shift_mode)
        self.is_train = bool(is_train)
        self.apply_shift_in_eval = bool(apply_shift_in_eval)

    def apply_crop_and_shift(self, image_tensor: torch.Tensor) -> torch.Tensor:
        _, height, width = image_tensor.shape
        start_y = (height - self.final_h) // 2
        start_x = (width - self.final_w) // 2

        shift_x = 0
        shift_y = 0
        if (self.is_train or self.apply_shift_in_eval) and self.shift != 0:
            if self.shift_mode == "fixed":
                shift_x = self.shift
                shift_y = self.shift
            elif self.shift_mode == "random":
                shift_x = torch.randint(-abs(self.shift), abs(self.shift) + 1, (1,)).item()
                shift_y = torch.randint(-abs(self.shift), abs(self.shift) + 1, (1,)).item()
            else:
                raise ValueError(f"Unknown shift mode: {self.shift_mode}")

        start_x = max(0, min(start_x + shift_x, width - self.final_w))
        start_y = max(0, min(start_y + shift_y, height - self.final_h))
        return image_tensor[:, start_y : start_y + self.final_h, start_x : start_x + self.final_w]


class SpeckleRealDataset(BaseSpeckleDataset):
    """Dataset for simulated propagated intensity fields."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        root_dir: Path,
        transform=None,
        canvas_size: tuple[int, int] = (256, 256),
        final_size: tuple[int, int] = (64, 64),
        shift: int = 0,
        shift_mode: str = "fixed",
        is_train: bool = True,
        apply_shift_in_eval: bool = False,
    ) -> None:
        super().__init__(transform, canvas_size, final_size, shift, shift_mode, is_train, apply_shift_in_eval)
        self.metadata = metadata.reset_index(drop=True)
        self.root_dir = Path(root_dir)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[idx]
        image = np.load(self.root_dir / row.iloc[0]).astype(np.float32)
        label = int(row.iloc[1]) - 1

        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        orig_h, orig_w = image.shape
        if (orig_h, orig_w) != (self.canvas_h, self.canvas_w):
            downsampler = torch.nn.AvgPool2d((orig_h // self.canvas_h, orig_w // self.canvas_w))
            tensor = downsampler(tensor)
        tensor = tensor.squeeze(0)
        tensor = self.apply_crop_and_shift(tensor)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, torch.tensor(label, dtype=torch.long)


class SpeckleGeneratedDataset(BaseSpeckleDataset):
    """Dataset for generated 256x256 synthetic intensity fields."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        root_dir: Path,
        transform=None,
        canvas_size: tuple[int, int] = (256, 256),
        final_size: tuple[int, int] = (64, 64),
        shift: int = 0,
        shift_mode: str = "fixed",
        is_train: bool = True,
        apply_shift_in_eval: bool = False,
        stage_folder: str = "stage5_pretrained_data",
    ) -> None:
        super().__init__(transform, canvas_size, final_size, shift, shift_mode, is_train, apply_shift_in_eval)
        self.metadata = metadata.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.stage_folder = str(stage_folder)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[idx]
        label = int(row["class_label"]) - 1
        class_dir = self.root_dir / f"class-{label + 1}" / self.stage_folder
        image = np.load(class_dir / row["filepath"]).astype(np.float32)
        tensor = torch.from_numpy(image).unsqueeze(0)
        tensor = self.apply_crop_and_shift(tensor)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, torch.tensor(label, dtype=torch.long)


def load_generated_metadata(gen_root: Path, stage_folder: str = "stage5_pretrained_data") -> pd.DataFrame:
    """Load generated-sample metadata from the pipeline output layout."""

    rows: list[dict[str, object]] = []
    for class_dir in sorted(Path(gen_root).glob("class-*")):
        if not class_dir.is_dir():
            continue
        class_label = int(class_dir.name.split("-", 1)[1])
        current_stage = class_dir / stage_folder
        if not current_stage.exists():
            continue
        for path in sorted(current_stage.glob("*.npy")):
            rows.append({"filepath": path.name, "class_label": class_label})
    return pd.DataFrame(rows, columns=["filepath", "class_label"])


def compute_dataset_stats(
    real_metadata: pd.DataFrame,
    real_root: Path,
    gen_metadata: pd.DataFrame | None,
    gen_root: Path | None,
    canvas_size: tuple[int, int],
    final_size: tuple[int, int],
    shift: int,
    shift_mode: str,
    use_acf: bool,
    batch_size: int = 64,
) -> tuple[float, float]:
    """Compute empirical mean and standard deviation after preprocessing."""

    transform = ACFTransform() if use_acf else None
    real_dataset = SpeckleRealDataset(
        real_metadata,
        real_root,
        transform=transform,
        canvas_size=canvas_size,
        final_size=final_size,
        shift=shift,
        shift_mode=shift_mode,
        is_train=False,
        apply_shift_in_eval=False,
    )

    if gen_metadata is not None and not gen_metadata.empty and gen_root is not None:
        gen_dataset = SpeckleGeneratedDataset(
            gen_metadata,
            gen_root,
            transform=transform,
            canvas_size=canvas_size,
            final_size=final_size,
            shift=shift,
            shift_mode=shift_mode,
            is_train=False,
            apply_shift_in_eval=False,
        )
        dataset = ConcatDataset([real_dataset, gen_dataset])
    else:
        dataset = real_dataset

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total_sum = 0.0
    total_sq = 0.0
    total_count = 0
    for images, _ in tqdm(loader, desc="Computing stats", leave=False):
        total_sum += torch.sum(images).item()
        total_sq += torch.sum(images**2).item()
        total_count += images.numel()

    mean = total_sum / total_count
    var = max(total_sq / total_count - mean**2, 0.0)
    return float(mean), float(np.sqrt(var))
