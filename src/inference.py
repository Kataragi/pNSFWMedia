#!/usr/bin/env python3
"""
pNSFWMedia Inference Script

Performs NSFW classification on images or pre-computed embeddings.
Supports both single-image and batch inference.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm


class NSFWClassifier:
    """NSFW classifier using pre-trained pNSFWMedia model."""

    def __init__(
        self,
        model_path: str,
        clip_extractor=None,
        threshold: float = 0.5
    ):
        """
        Initialize the NSFW classifier.

        Args:
            model_path: Path to the trained Keras model
            clip_extractor: Optional CLIPEmbeddingExtractor for image inference
            threshold: Classification threshold (default: 0.5)
        """
        self.model = tf.keras.models.load_model(model_path)
        self.clip_extractor = clip_extractor
        self.threshold = threshold

        print(f"Loaded model from: {model_path}")
        print(f"Classification threshold: {threshold}")

    def predict_from_embedding(self, embedding: np.ndarray) -> dict:
        """
        Predict NSFW probability from a 256-dim embedding.

        Args:
            embedding: numpy array of shape (256,) or (batch_size, 256)

        Returns:
            Dictionary with probability and classification
        """
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        prob = self.model.predict(embedding, verbose=0)
        prob = prob.flatten()

        if len(prob) == 1:
            return {
                'nsfw_probability': float(prob[0]),
                'is_nsfw': bool(prob[0] >= self.threshold),
                'classification': 'nsfw' if prob[0] >= self.threshold else 'sfw'
            }
        else:
            return {
                'nsfw_probabilities': prob.tolist(),
                'classifications': ['nsfw' if p >= self.threshold else 'sfw' for p in prob]
            }

    def predict_from_image(self, image_path: str) -> dict:
        """
        Predict NSFW probability from an image file.

        Requires clip_extractor to be set.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with probability and classification
        """
        if self.clip_extractor is None:
            raise ValueError("CLIPEmbeddingExtractor is required for image inference")

        embedding = self.clip_extractor.extract_embedding(image_path)
        result = self.predict_from_embedding(embedding)
        result['image_path'] = image_path

        return result

    def predict_from_npy(self, npy_path: str) -> dict:
        """
        Predict NSFW probability from a .npy embedding file.

        Args:
            npy_path: Path to the .npy file

        Returns:
            Dictionary with probability and classification
        """
        embedding = np.load(npy_path)
        result = self.predict_from_embedding(embedding)
        result['embedding_path'] = npy_path

        return result

    def batch_predict_from_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Batch predict NSFW probabilities.

        Args:
            embeddings: numpy array of shape (batch_size, 256)

        Returns:
            numpy array of probabilities
        """
        return self.model.predict(embeddings, verbose=0).flatten()


def inference_from_embedding_directory(
    classifier: NSFWClassifier,
    input_dir: str,
    output_path: str = None,
    batch_size: int = 256
) -> list:
    """
    Run inference on all .npy files in a directory.

    Args:
        classifier: NSFWClassifier instance
        input_dir: Directory containing .npy embedding files
        output_path: Optional path to save results as JSON
        batch_size: Batch size for processing

    Returns:
        List of prediction results
    """
    input_path = Path(input_dir)
    npy_files = sorted(input_path.glob("**/*.npy"))

    if not npy_files:
        print(f"No .npy files found in {input_dir}")
        return []

    print(f"Found {len(npy_files)} embedding files")

    results = []

    # Process in batches
    for i in tqdm(range(0, len(npy_files), batch_size), desc="Predicting"):
        batch_files = npy_files[i:i + batch_size]

        # Load embeddings
        embeddings = []
        valid_files = []
        for npy_file in batch_files:
            try:
                emb = np.load(npy_file)
                if emb.shape == (256,):
                    embeddings.append(emb)
                    valid_files.append(npy_file)
            except Exception as e:
                print(f"Error loading {npy_file}: {e}")

        if not embeddings:
            continue

        # Batch predict
        embeddings = np.array(embeddings)
        probs = classifier.batch_predict_from_embeddings(embeddings)

        # Store results
        for npy_file, prob in zip(valid_files, probs):
            results.append({
                'file': str(npy_file),
                'name': npy_file.stem,
                'nsfw_probability': float(prob),
                'is_nsfw': bool(prob >= classifier.threshold),
                'classification': 'nsfw' if prob >= classifier.threshold else 'sfw'
            })

    # Summary statistics
    n_nsfw = sum(1 for r in results if r['is_nsfw'])
    n_sfw = len(results) - n_nsfw
    avg_prob = np.mean([r['nsfw_probability'] for r in results])

    print(f"\nResults Summary:")
    print(f"  Total: {len(results)}")
    print(f"  NSFW:  {n_nsfw} ({100 * n_nsfw / len(results):.1f}%)")
    print(f"  SFW:   {n_sfw} ({100 * n_sfw / len(results):.1f}%)")
    print(f"  Avg probability: {avg_prob:.4f}")

    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                'summary': {
                    'total': len(results),
                    'nsfw': n_nsfw,
                    'sfw': n_sfw,
                    'avg_probability': avg_prob,
                    'threshold': classifier.threshold
                },
                'predictions': results
            }, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return results


def inference_from_image_directory(
    classifier: NSFWClassifier,
    input_dir: str,
    output_path: str = None
) -> list:
    """
    Run inference on all images in a directory.

    Requires classifier to have clip_extractor set.

    Args:
        classifier: NSFWClassifier instance with clip_extractor
        input_dir: Directory containing images
        output_path: Optional path to save results as JSON

    Returns:
        List of prediction results
    """
    if classifier.clip_extractor is None:
        raise ValueError("CLIPEmbeddingExtractor is required for image inference")

    input_path = Path(input_dir)
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

    image_files = [
        f for f in input_path.glob("**/*")
        if f.is_file() and f.suffix.lower() in extensions
    ]

    if not image_files:
        print(f"No images found in {input_dir}")
        return []

    print(f"Found {len(image_files)} images")

    results = []

    for image_file in tqdm(image_files, desc="Predicting"):
        try:
            result = classifier.predict_from_image(str(image_file))
            result['file'] = str(image_file)
            result['name'] = image_file.stem
            results.append(result)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    # Summary
    n_nsfw = sum(1 for r in results if r['is_nsfw'])
    n_sfw = len(results) - n_nsfw
    avg_prob = np.mean([r['nsfw_probability'] for r in results])

    print(f"\nResults Summary:")
    print(f"  Total: {len(results)}")
    print(f"  NSFW:  {n_nsfw} ({100 * n_nsfw / len(results):.1f}%)")
    print(f"  SFW:   {n_sfw} ({100 * n_sfw / len(results):.1f}%)")
    print(f"  Avg probability: {avg_prob:.4f}")

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                'summary': {
                    'total': len(results),
                    'nsfw': n_nsfw,
                    'sfw': n_sfw,
                    'avg_probability': avg_prob,
                    'threshold': classifier.threshold
                },
                'predictions': results
            }, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="pNSFWMedia inference on images or embeddings"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/pnsfwmedia_classifier.keras",
        help="Path to the trained Keras model"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--embedding",
        type=str,
        help="Path to a single .npy embedding file"
    )
    input_group.add_argument(
        "--embedding-dir",
        type=str,
        help="Directory containing .npy embedding files"
    )
    input_group.add_argument(
        "--image",
        type=str,
        help="Path to a single image file"
    )
    input_group.add_argument(
        "--image-dir",
        type=str,
        help="Directory containing images"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Path to save results as JSON"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for processing"
    )

    # CLIP options (for image inference)
    parser.add_argument(
        "--clip-model",
        type=str,
        default="ViT-B/32",
        help="CLIP model variant"
    )
    parser.add_argument(
        "--projection-path",
        type=str,
        default="models/clip_projection.pt",
        help="Path to CLIP projection layer weights"
    )

    args = parser.parse_args()

    # Initialize CLIP extractor if needed for image inference
    clip_extractor = None
    if args.image or args.image_dir:
        try:
            from extract_embeddings import CLIPEmbeddingExtractor
            clip_extractor = CLIPEmbeddingExtractor(
                model_name=args.clip_model,
                output_dim=256,
                projection_path=args.projection_path
            )
        except ImportError:
            # Try relative import
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from extract_embeddings import CLIPEmbeddingExtractor
            clip_extractor = CLIPEmbeddingExtractor(
                model_name=args.clip_model,
                output_dim=256,
                projection_path=args.projection_path
            )

    # Initialize classifier
    classifier = NSFWClassifier(
        model_path=args.model_path,
        clip_extractor=clip_extractor,
        threshold=args.threshold
    )

    # Run inference
    if args.embedding:
        result = classifier.predict_from_npy(args.embedding)
        print(f"\nPrediction for: {args.embedding}")
        print(f"  NSFW Probability: {result['nsfw_probability']:.4f}")
        print(f"  Classification: {result['classification']}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)

    elif args.embedding_dir:
        inference_from_embedding_directory(
            classifier,
            args.embedding_dir,
            args.output,
            args.batch_size
        )

    elif args.image:
        result = classifier.predict_from_image(args.image)
        print(f"\nPrediction for: {args.image}")
        print(f"  NSFW Probability: {result['nsfw_probability']:.4f}")
        print(f"  Classification: {result['classification']}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)

    elif args.image_dir:
        inference_from_image_directory(
            classifier,
            args.image_dir,
            args.output
        )


if __name__ == "__main__":
    main()
