#!/usr/bin/env python3
"""
Dataset Preparation Utility for pNSFWMedia

Organizes images into train/val splits with proper directory structure.
Supports both copying/moving files and creating symbolic links.
"""

import argparse
import os
import random
import shutil
from pathlib import Path

from tqdm import tqdm


def split_files(files: list, train_ratio: float = 0.8, seed: int = 42) -> tuple:
    """
    Split files into train and validation sets.

    Args:
        files: List of file paths
        train_ratio: Ratio of files for training (default: 0.8)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_files, val_files)
    """
    random.seed(seed)
    files = list(files)
    random.shuffle(files)

    split_idx = int(len(files) * train_ratio)
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    return train_files, val_files


def organize_dataset(
    sfw_dir: str,
    nsfw_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    mode: str = "copy",
    seed: int = 42
):
    """
    Organize images into train/val/sfw/nsfw structure.

    Args:
        sfw_dir: Directory containing SFW images
        nsfw_dir: Directory containing NSFW images
        output_dir: Output directory
        train_ratio: Train/val split ratio
        mode: 'copy', 'move', or 'symlink'
        seed: Random seed
    """
    output_path = Path(output_dir)
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

    # Create directory structure
    for split in ["train", "val"]:
        for cls in ["sfw", "nsfw"]:
            (output_path / split / cls).mkdir(parents=True, exist_ok=True)

    # Process each class
    for cls, src_dir in [("sfw", sfw_dir), ("nsfw", nsfw_dir)]:
        src_path = Path(src_dir)

        if not src_path.exists():
            print(f"Warning: {src_path} does not exist, skipping")
            continue

        # Get all image files
        files = [
            f for f in src_path.glob("*")
            if f.is_file() and f.suffix.lower() in extensions
        ]

        print(f"\nProcessing {cls}: {len(files)} images")

        # Split files
        train_files, val_files = split_files(files, train_ratio, seed)
        print(f"  Train: {len(train_files)}, Val: {len(val_files)}")

        # Copy/move files
        for split, split_files in [("train", train_files), ("val", val_files)]:
            dest_dir = output_path / split / cls

            for src_file in tqdm(split_files, desc=f"  {split}/{cls}"):
                dest_file = dest_dir / src_file.name

                if mode == "copy":
                    shutil.copy2(src_file, dest_file)
                elif mode == "move":
                    shutil.move(src_file, dest_file)
                elif mode == "symlink":
                    if dest_file.exists():
                        dest_file.unlink()
                    dest_file.symlink_to(src_file.absolute())

    # Print summary
    print("\n" + "=" * 50)
    print("Dataset Summary")
    print("=" * 50)

    total = 0
    for split in ["train", "val"]:
        print(f"\n{split.upper()}:")
        for cls in ["sfw", "nsfw"]:
            cls_dir = output_path / split / cls
            count = len(list(cls_dir.glob("*")))
            print(f"  {cls}: {count}")
            total += count

    print(f"\nTotal: {total} images")
    print(f"Output directory: {output_path}")


def verify_dataset(dataset_dir: str):
    """
    Verify the dataset structure and print statistics.

    Args:
        dataset_dir: Dataset directory
    """
    dataset_path = Path(dataset_dir)
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".npy"}

    print("=" * 50)
    print("Dataset Verification")
    print("=" * 50)

    for split in ["train", "val"]:
        split_path = dataset_path / split
        if not split_path.exists():
            print(f"\n{split.upper()}: NOT FOUND")
            continue

        print(f"\n{split.upper()}:")

        for cls in ["sfw", "nsfw"]:
            cls_path = split_path / cls
            if not cls_path.exists():
                print(f"  {cls}: NOT FOUND")
                continue

            files = [
                f for f in cls_path.glob("*")
                if f.is_file() and f.suffix.lower() in extensions
            ]
            print(f"  {cls}: {len(files)}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for pNSFWMedia training"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Organize command
    organize_parser = subparsers.add_parser(
        "organize",
        help="Organize images into train/val structure"
    )
    organize_parser.add_argument(
        "--sfw-dir",
        type=str,
        required=True,
        help="Directory containing SFW images"
    )
    organize_parser.add_argument(
        "--nsfw-dir",
        type=str,
        required=True,
        help="Directory containing NSFW images"
    )
    organize_parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/images",
        help="Output directory"
    )
    organize_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train/val split ratio (default: 0.8)"
    )
    organize_parser.add_argument(
        "--mode",
        type=str,
        default="copy",
        choices=["copy", "move", "symlink"],
        help="How to handle files"
    )
    organize_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    # Verify command
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify dataset structure"
    )
    verify_parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset/images",
        help="Dataset directory to verify"
    )

    args = parser.parse_args()

    if args.command == "organize":
        organize_dataset(
            sfw_dir=args.sfw_dir,
            nsfw_dir=args.nsfw_dir,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            mode=args.mode,
            seed=args.seed
        )

    elif args.command == "verify":
        verify_dataset(args.dataset_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
