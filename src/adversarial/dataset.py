"""
Dataset module for adversarial perturbation training.

Provides paired SFW/NSFW image loading so that the perturbation generator
can learn from the feature-space difference between the two classes.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class NSFWSFWImageDataset(Dataset):
    """
    Dataset that yields (nsfw_image, sfw_image) pairs.

    For each NSFW image, a random SFW image is sampled as a reference.
    This pairing enables the model to learn the NSFW-to-SFW feature gap.

    Directory layout expected:
        root/
          nsfw/
            img001.jpg
            ...
          sfw/
            img001.jpg
            ...
    """

    def __init__(
        self,
        root_dir: str,
        image_size: int = 224,
        augment: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            root_dir: Path containing ``nsfw/`` and ``sfw/`` subdirectories.
            image_size: Resize target (square).
            augment: Apply random horizontal flip during training.
            seed: Random seed for reproducibility.
        """
        self.image_size = image_size
        self.augment = augment
        self.rng = random.Random(seed)

        nsfw_dir = Path(root_dir) / "nsfw"
        sfw_dir = Path(root_dir) / "sfw"

        if not nsfw_dir.exists():
            raise FileNotFoundError(f"NSFW directory not found: {nsfw_dir}")
        if not sfw_dir.exists():
            raise FileNotFoundError(f"SFW directory not found: {sfw_dir}")

        self.nsfw_files = sorted(
            [f for f in nsfw_dir.iterdir()
             if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
        )
        self.sfw_files = sorted(
            [f for f in sfw_dir.iterdir()
             if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
        )

        if not self.nsfw_files:
            raise ValueError(f"No NSFW images found in {nsfw_dir}")
        if not self.sfw_files:
            raise ValueError(f"No SFW images found in {sfw_dir}")

        print(f"[Dataset] NSFW images: {len(self.nsfw_files)}")
        print(f"[Dataset] SFW images:  {len(self.sfw_files)}")

    def __len__(self):
        return len(self.nsfw_files)

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load image, resize, convert to [0,1] float tensor [3, H, W]."""
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BICUBIC)

        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW

        if self.augment and random.random() > 0.5:
            tensor = torch.flip(tensor, dims=[2])  # horizontal flip

        return tensor

    def __getitem__(self, idx):
        nsfw_img = self._load_image(self.nsfw_files[idx])

        # Random SFW reference
        sfw_idx = self.rng.randint(0, len(self.sfw_files) - 1)
        sfw_img = self._load_image(self.sfw_files[sfw_idx])

        return {
            "nsfw_image": nsfw_img,
            "sfw_image": sfw_img,
            "nsfw_path": str(self.nsfw_files[idx]),
            "sfw_path": str(self.sfw_files[sfw_idx]),
        }


class EmbeddingOnlyDataset(Dataset):
    """
    Fallback dataset that loads pre-computed 256-dim embeddings
    (for when raw images are not available but .npy embeddings exist).

    This can be used for a lightweight embedding-space-only attack variant.
    """

    def __init__(self, embeddings_dir: str, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim

        nsfw_dir = Path(embeddings_dir) / "nsfw"
        sfw_dir = Path(embeddings_dir) / "sfw"

        self.nsfw_files = sorted(nsfw_dir.glob("*.npy")) if nsfw_dir.exists() else []
        self.sfw_files = sorted(sfw_dir.glob("*.npy")) if sfw_dir.exists() else []

        if not self.nsfw_files:
            raise ValueError(f"No NSFW embeddings found in {nsfw_dir}")
        if not self.sfw_files:
            raise ValueError(f"No SFW embeddings found in {sfw_dir}")

        # Pre-load all SFW embeddings for fast centroid computation
        sfw_embs = []
        for f in self.sfw_files:
            emb = np.load(f)
            if emb.shape == (embedding_dim,):
                sfw_embs.append(emb)
        self.sfw_embeddings = np.array(sfw_embs, dtype=np.float32)

        print(f"[EmbeddingDataset] NSFW: {len(self.nsfw_files)}, "
              f"SFW: {len(self.sfw_files)}")

    def __len__(self):
        return len(self.nsfw_files)

    def __getitem__(self, idx):
        nsfw_emb = np.load(self.nsfw_files[idx]).astype(np.float32)
        sfw_idx = random.randint(0, len(self.sfw_files) - 1)
        sfw_emb = np.load(self.sfw_files[sfw_idx]).astype(np.float32)

        return {
            "nsfw_embedding": torch.from_numpy(nsfw_emb),
            "sfw_embedding": torch.from_numpy(sfw_emb),
        }


def create_dataloader(
    root_dir: str,
    batch_size: int = 8,
    image_size: int = 224,
    num_workers: int = 4,
    augment: bool = True,
    seed: int = 42,
) -> DataLoader:
    """
    Create a DataLoader for adversarial training.

    Args:
        root_dir: Path with nsfw/ and sfw/ subdirectories of images.
        batch_size: Batch size.
        image_size: Image resize target.
        num_workers: DataLoader workers.
        augment: Whether to augment.
        seed: Random seed.

    Returns:
        DataLoader yielding dicts with ``nsfw_image`` and ``sfw_image`` tensors.
    """
    dataset = NSFWSFWImageDataset(
        root_dir=root_dir,
        image_size=image_size,
        augment=augment,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
