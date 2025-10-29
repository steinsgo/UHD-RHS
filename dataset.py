import os
import random
from typing import Dict, Iterable, List, Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


DEFAULT_MEAN = [0.45837133, 0.47633536, 0.44432645]
DEFAULT_STD = [0.16936361, 0.15927625, 0.15468806]


class PairedRestorationDataset(Dataset):
    """Generic paired dataset where degraded images live in sub-folders.

    Each item returns a degraded image tensor, the clean target, and metadata
    indicating which degradation bucket the sample originated from.
    """

    def __init__(
        self,
        root_dir: str,
        degrade_folders: Iterable[str],
        normalize: bool = False,
        crop_size: int = 0,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        random_degrade: bool = True,
    ) -> None:
        self.root_dir = root_dir
        self.degrade_folders = list(degrade_folders)
        if len(self.degrade_folders) == 0:
            raise ValueError("degrade_folders must contain at least one entry")

        self.clear_dir = os.path.join(self.root_dir, "clear")
        if not os.path.isdir(self.clear_dir):
            raise FileNotFoundError(f"Missing clear directory: {self.clear_dir}")

        self.image_filenames = sorted(
            [
                img
                for img in os.listdir(self.clear_dir)
                if img.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        if len(self.image_filenames) == 0:
            raise FileNotFoundError(f"No images found in {self.clear_dir}")

        self.crop_size = crop_size
        self.random_degrade = random_degrade
        mean = mean or DEFAULT_MEAN
        std = std or DEFAULT_STD
        transform_list = [transforms.ToTensor()]
        if normalize:
            transform_list.append(transforms.Normalize(mean, std))
        self.transform = transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.image_filenames)

    def _pick_degrade_folder(self, idx: int) -> str:
        if len(self.degrade_folders) == 1:
            return self.degrade_folders[0]
        if self.random_degrade:
            return random.choice(self.degrade_folders)
        offset = idx % len(self.degrade_folders)
        return self.degrade_folders[offset]

    def _load_pair(self, degrade_path: str, clear_path: str) -> tuple[Image.Image, Image.Image]:
        degraded = Image.open(degrade_path).convert("RGB")
        clear = Image.open(clear_path).convert("RGB")
        if degraded.size != clear.size:
            raise ValueError("The size of the input image pairs is inconsistent")
        return degraded, clear

    def _random_crop(self, degraded: Image.Image, clear: Image.Image) -> tuple[Image.Image, Image.Image]:
        if self.crop_size is None or self.crop_size <= 0:
            return degraded, clear
        if self.crop_size > clear.width or self.crop_size > clear.height:
            raise ValueError("Crop size is larger than the image size")
        start_x = random.randint(0, clear.width - self.crop_size)
        start_y = random.randint(0, clear.height - self.crop_size)
        crop_box = (start_x, start_y, start_x + self.crop_size, start_y + self.crop_size)
        return degraded.crop(crop_box), clear.crop(crop_box)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        filename = self.image_filenames[idx]
        degrade_folder = self._pick_degrade_folder(idx)
        degrade_dir = os.path.join(self.root_dir, degrade_folder)
        if not os.path.isdir(degrade_dir):
            raise FileNotFoundError(f"Missing degrade directory: {degrade_dir}")
        degrade_path = os.path.join(degrade_dir, filename)
        clear_path = os.path.join(self.clear_dir, filename)
        if not os.path.exists(degrade_path):
            raise FileNotFoundError(f"Missing degraded image: {degrade_path}")
        degraded, clear = self._load_pair(degrade_path, clear_path)
        degraded, clear = self._random_crop(degraded, clear)

        random_rotate = random.choice([0, 90, 180, 270])
        degraded = degraded.rotate(random_rotate)
        clear = clear.rotate(random_rotate)

        degraded_tensor = self.transform(degraded)
        clear_tensor = self.transform(clear)
        return {
            "cloud_img": degraded_tensor,
            "input_img": degraded_tensor,
            "clear_img": clear_tensor,
            "degrade_folder": degrade_folder,
            "task": degrade_folder,
            "filename": filename,
        }


class CloudRemovalDataset(PairedRestorationDataset):
    """8K Dehaze dataset with cloud_L1...cloud_L4 degradations."""

    def __init__(
        self,
        root_dir: str,
        normalize: bool,
        crop_size: int,
        levels: Optional[Iterable[int]] = None,
    ) -> None:
        levels = levels or [1, 2, 3, 4]
        folders = [f"cloud_L{lvl}" for lvl in levels]
        super().__init__(root_dir, folders, normalize=normalize, crop_size=crop_size, random_degrade=True)


class RainRemovalDataset(PairedRestorationDataset):
    """Paired rain removal dataset with clear/addrain structure."""

    def __init__(
        self,
        root_dir: str,
        normalize: bool,
        crop_size: int,
        degrade_folder: str = "addrain",
    ) -> None:
        super().__init__(root_dir, [degrade_folder], normalize=normalize, crop_size=crop_size, random_degrade=False)


class SnowRemovalDataset(PairedRestorationDataset):
    """Paired snow removal dataset (clear/addsnow or similar)."""

    def __init__(
        self,
        root_dir: str,
        normalize: bool,
        crop_size: int,
        degrade_folder: str = "addsnow",
    ) -> None:
        super().__init__(root_dir, [degrade_folder], normalize=normalize, crop_size=crop_size, random_degrade=False)


class TaskLabeledDataset(Dataset):
    """Wrap another dataset and override the task label."""

    def __init__(self, dataset: Dataset, task_label: str) -> None:
        self.dataset = dataset
        self.task_label = task_label

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.dataset[idx]
        sample["task"] = self.task_label
        return sample


class CompositeRestorationDataset(Dataset):
    """Sample across multiple degradation datasets with optional weighting."""

    def __init__(
        self,
        datasets: Dict[str, PairedRestorationDataset],
        sampling_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        if len(datasets) == 0:
            raise ValueError("Provide at least one dataset")
        self.datasets = datasets
        weights = sampling_weights or {k: 1.0 for k in datasets}
        total = sum(max(v, 0.0) for v in weights.values())
        if total <= 0:
            raise ValueError("Sampling weights must sum to a positive value")
        self.normalized_weights = {k: max(v, 0.0) / total for k, v in weights.items()}
        self.tasks = list(self.normalized_weights.keys())

    def __len__(self) -> int:
        return sum(len(ds) for ds in self.datasets.values())

    def _sample_task(self) -> str:
        r = random.random()
        cumulative = 0.0
        for task in self.tasks:
            cumulative += self.normalized_weights[task]
            if r <= cumulative:
                return task
        return self.tasks[-1]

    def __getitem__(self, idx: int) -> Dict[str, object]:
        task = self._sample_task()
        ds = self.datasets[task]
        sample = ds[random.randrange(len(ds))]
        sample["task"] = task
        return sample


if __name__ == "__main__":
    dataset = CloudRemovalDataset(r"./datasets/8KDehaze_mini", False, crop_size=512)
    print(dataset[0]["cloud_img"].shape, dataset[0]["degrade_folder"])
