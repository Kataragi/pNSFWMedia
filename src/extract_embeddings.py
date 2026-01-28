#!/usr/bin/env python3
"""
Stage A: CLIP Embedding Extractor for pNSFWMedia

Extracts image embeddings using OpenAI CLIP and projects to 256 dimensions.
Embeddings are cached as .npy files for efficient training.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

try:
    import clip
except ImportError:
    print("Please install CLIP: pip install git+https://github.com/openai/CLIP.git")
    raise


class CLIPEmbeddingExtractor:
    """
    CLIP-based image embedding extractor with linear projection to 256 dimensions.

    Uses OpenAI CLIP (ViT-B/32) as the backbone and projects the 512-dim
    output to 256 dimensions to match the pNSFWMedia specification.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        output_dim: int = 256,
        device: str = None,
        projection_path: str = None
    ):
        """
        Initialize the CLIP embedding extractor.

        Args:
            model_name: CLIP model variant (default: ViT-B/32)
            output_dim: Target embedding dimension (default: 256)
            device: Device to run inference on (default: auto-detect)
            projection_path: Path to save/load projection layer weights
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dim = output_dim

        print(f"Loading CLIP model: {model_name} on {self.device}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        # Get CLIP output dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224).to(self.device)
            clip_output = self.model.encode_image(dummy_input)
            self.clip_dim = clip_output.shape[-1]

        print(f"CLIP output dimension: {self.clip_dim}")
        print(f"Target output dimension: {self.output_dim}")

        # Linear projection layer: clip_dim -> 256
        self.projection = nn.Linear(self.clip_dim, self.output_dim, bias=False).to(self.device)

        # Initialize with random orthogonal weights for better performance
        nn.init.orthogonal_(self.projection.weight)

        # Load or save projection weights
        if projection_path and os.path.exists(projection_path):
            print(f"Loading projection weights from: {projection_path}")
            self.projection.load_state_dict(torch.load(projection_path, map_location=self.device))
        elif projection_path:
            print(f"Saving initial projection weights to: {projection_path}")
            os.makedirs(os.path.dirname(projection_path), exist_ok=True)
            torch.save(self.projection.state_dict(), projection_path)

    def extract_embedding(self, image_path: str) -> np.ndarray:
        """
        Extract 256-dimensional embedding from a single image.

        Args:
            image_path: Path to the image file

        Returns:
            numpy array of shape (256,)
        """
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Get CLIP embedding
            clip_embedding = self.model.encode_image(image_tensor)
            clip_embedding = clip_embedding.float()

            # Project to 256 dimensions
            projected = self.projection(clip_embedding)

            # L2 normalize
            projected = projected / projected.norm(dim=-1, keepdim=True)

        return projected.cpu().numpy().squeeze()

    def extract_batch(self, image_paths: list) -> np.ndarray:
        """
        Extract embeddings from a batch of images.

        Args:
            image_paths: List of image file paths

        Returns:
            numpy array of shape (batch_size, 256)
        """
        images = []
        for path in image_paths:
            try:
                image = Image.open(path).convert("RGB")
                images.append(self.preprocess(image))
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue

        if not images:
            return np.array([])

        batch_tensor = torch.stack(images).to(self.device)

        with torch.no_grad():
            clip_embeddings = self.model.encode_image(batch_tensor)
            clip_embeddings = clip_embeddings.float()
            projected = self.projection(clip_embeddings)
            projected = projected / projected.norm(dim=-1, keepdim=True)

        return projected.cpu().numpy()


def extract_embeddings_from_directory(
    extractor: CLIPEmbeddingExtractor,
    input_dir: str,
    output_dir: str,
    batch_size: int = 32
):
    """
    Extract embeddings from all images in a directory and save as .npy files.

    Args:
        extractor: CLIPEmbeddingExtractor instance
        input_dir: Directory containing images
        output_dir: Directory to save embeddings
        batch_size: Batch size for processing
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Supported image extensions
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

    # Get all image files
    image_files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ]

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images in {input_dir}")

    # Process in batches
    for i in tqdm(range(0, len(image_files), batch_size), desc=f"Processing {input_dir}"):
        batch_files = image_files[i:i + batch_size]
        batch_paths = [str(f) for f in batch_files]

        embeddings = extractor.extract_batch(batch_paths)

        # Save each embedding
        for j, (file_path, embedding) in enumerate(zip(batch_files, embeddings)):
            output_file = output_path / f"{file_path.stem}.npy"
            np.save(output_file, embedding.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(
        description="Extract CLIP embeddings for pNSFWMedia training"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="dataset/images",
        help="Root directory containing train/val/sfw/nsfw subdirectories"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/embeddings",
        help="Root directory to save embeddings"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B/32",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101"],
        help="CLIP model variant"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--projection-path",
        type=str,
        default="models/clip_projection.pt",
        help="Path to save/load projection layer weights"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)"
    )

    args = parser.parse_args()

    # Initialize extractor
    extractor = CLIPEmbeddingExtractor(
        model_name=args.model,
        output_dim=256,
        device=args.device,
        projection_path=args.projection_path
    )

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    # Process each split and class
    splits = ["train", "val"]
    classes = ["sfw", "nsfw"]

    for split in splits:
        for cls in classes:
            input_dir = input_root / split / cls
            output_dir = output_root / split / cls

            if input_dir.exists():
                print(f"\nProcessing {split}/{cls}...")
                extract_embeddings_from_directory(
                    extractor,
                    str(input_dir),
                    str(output_dir),
                    batch_size=args.batch_size
                )
            else:
                print(f"Skipping {input_dir} (not found)")

    print("\nEmbedding extraction complete!")
    print(f"Embeddings saved to: {output_root}")


if __name__ == "__main__":
    main()
