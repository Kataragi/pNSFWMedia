#!/usr/bin/env python3
"""
Stage A: NudeNet Embedding Extractor for pNSFWMedia

Extracts image embeddings using NudeNet's YOLOv8 ONNX backbone and projects
to 256 dimensions via random orthogonal projection.
Embeddings are cached as .npy files for efficient training.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from tqdm import tqdm


def _find_nudenet_model_path() -> str:
    """Locate the NudeNet ONNX model file."""
    try:
        import nudenet
        package_dir = Path(nudenet.__file__).parent
        model_path = package_dir / "best.onnx"
        if model_path.exists():
            return str(model_path)
    except ImportError:
        pass

    # Fallback: search common locations
    candidates = [
        Path.home() / ".nudenet" / "best.onnx",
        Path("models") / "best.onnx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "NudeNet ONNX model not found. Install nudenet: pip install nudenet"
    )


def _find_backbone_node(model: onnx.ModelProto) -> str:
    """
    Heuristic to find the backbone feature node in a YOLOv8 ONNX model.

    Strategy: Find the last Conv output that is consumed by a Concat node,
    which typically marks the boundary between backbone and FPN/neck.
    """
    # Build a map of which nodes consume each output
    consumers = {}
    for node in model.graph.node:
        for inp in node.input:
            if inp not in consumers:
                consumers[inp] = []
            consumers[inp].append(node.op_type)

    # Find Conv nodes whose output feeds into Concat (backbone → neck boundary)
    candidates = []
    for node in model.graph.node:
        if node.op_type == "Conv" and len(node.output) > 0:
            output_name = node.output[0]
            if output_name in consumers:
                if "Concat" in consumers[output_name]:
                    candidates.append(output_name)

    if candidates:
        # Return the last such node (deepest backbone feature)
        return candidates[-1]

    # Fallback: last Conv node output
    conv_outputs = []
    for node in model.graph.node:
        if node.op_type == "Conv" and len(node.output) > 0:
            conv_outputs.append(node.output[0])

    if conv_outputs:
        return conv_outputs[-1]

    raise RuntimeError("Could not find a suitable backbone feature node in the ONNX model")


def _list_onnx_nodes(model_path: str):
    """Print all nodes in the ONNX model for debugging."""
    model = onnx.load(model_path)
    print(f"Model: {model_path}")
    print(f"IR version: {model.ir_version}")
    print(f"Opset: {[op.version for op in model.opset_import]}")
    print(f"\nInputs:")
    for inp in model.graph.input:
        shape = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {shape}")
    print(f"\nOutputs:")
    for out in model.graph.output:
        shape = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {shape}")
    print(f"\nNodes ({len(model.graph.node)} total):")
    for i, node in enumerate(model.graph.node):
        print(f"  [{i:4d}] {node.op_type:20s} -> {list(node.output)}")


class NudeNetEmbeddingExtractor:
    """
    NudeNet-based image embedding extractor with orthogonal projection to 256 dims.

    Uses NudeNet's YOLOv8 ONNX backbone to extract feature maps, applies
    Global Average Pooling, and projects to 256 dimensions.
    """

    def __init__(
        self,
        output_dim: int = 256,
        model_path: str = None,
        feature_node: str = None,
        projection_path: str = None,
    ):
        """
        Initialize the NudeNet embedding extractor.

        Args:
            output_dim: Target embedding dimension (default: 256)
            model_path: Path to NudeNet ONNX model (default: auto-detect)
            feature_node: ONNX node name for feature extraction (default: auto-detect)
            projection_path: Path to save/load projection matrix
        """
        self.output_dim = output_dim
        self.model_path = model_path or _find_nudenet_model_path()

        # Load and modify ONNX model to expose backbone features
        onnx_model = onnx.load(self.model_path)
        self.feature_node_name = feature_node or _find_backbone_node(onnx_model)
        print(f"Feature extraction node: {self.feature_node_name}")

        # Add the intermediate node as an output
        intermediate_output = onnx.helper.make_tensor_value_info(
            self.feature_node_name, onnx.TensorProto.FLOAT, None
        )
        onnx_model.graph.output.append(intermediate_output)

        # Serialize modified model
        modified_bytes = onnx_model.SerializeToString()

        # Setup ONNX Runtime session with CUDA auto-detection
        providers = []
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
            print("[GPU] CUDA available for NudeNet inference")
        providers.append("CPUExecutionProvider")

        if not any(p != "CPUExecutionProvider" for p in providers):
            print("[CPU] Using CPU for NudeNet inference")

        self.session = ort.InferenceSession(modified_bytes, providers=providers)

        # Get model input shape
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        self.input_shape = input_info.shape  # e.g. [1, 3, 320, 320]
        self.input_h = self.input_shape[2] if isinstance(self.input_shape[2], int) else 320
        self.input_w = self.input_shape[3] if isinstance(self.input_shape[3], int) else 320
        print(f"Model input: {self.input_name} shape={self.input_shape}")

        # Determine backbone feature dimension by running a dummy input
        dummy = np.zeros((1, 3, self.input_h, self.input_w), dtype=np.float32)
        outputs = self.session.run([self.feature_node_name], {self.input_name: dummy})
        feature_shape = outputs[0].shape
        print(f"Backbone feature shape: {feature_shape}")

        # Feature dim after Global Average Pooling
        self.feature_dim = feature_shape[1]  # channels dimension (N, C, H, W)
        print(f"Feature dimension (after GAP): {self.feature_dim}")
        print(f"Target output dimension: {self.output_dim}")

        # Initialize or load projection matrix
        self._init_projection(projection_path)

    def _init_projection(self, projection_path: str = None):
        """Initialize random orthogonal projection matrix."""
        if projection_path and os.path.exists(projection_path):
            print(f"Loading projection matrix from: {projection_path}")
            self.projection = np.load(projection_path)
            if self.projection.shape != (self.feature_dim, self.output_dim):
                print(
                    f"Warning: projection shape mismatch "
                    f"{self.projection.shape} vs expected "
                    f"({self.feature_dim}, {self.output_dim}). Reinitializing."
                )
                self.projection = self._create_orthogonal_projection()
        else:
            self.projection = self._create_orthogonal_projection()

        if projection_path:
            os.makedirs(os.path.dirname(projection_path) or ".", exist_ok=True)
            np.save(projection_path, self.projection)
            print(f"Projection matrix saved to: {projection_path}")

    def _create_orthogonal_projection(self) -> np.ndarray:
        """Create a random orthogonal projection matrix."""
        rng = np.random.default_rng(seed=42)
        random_matrix = rng.standard_normal((self.feature_dim, self.output_dim))
        # QR decomposition for orthogonal matrix
        q, _ = np.linalg.qr(random_matrix)
        # If feature_dim < output_dim, q may be smaller; handle gracefully
        if q.shape[1] < self.output_dim:
            pad = rng.standard_normal((self.feature_dim, self.output_dim - q.shape[1]))
            q = np.hstack([q, pad])
        return q[:, :self.output_dim].astype(np.float32)

    def _preprocess(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for NudeNet ONNX model.

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed array of shape (3, H, W) as float32
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_w, self.input_h))
        img = img.astype(np.float32) / 255.0
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        return img

    def _features_to_embedding(self, features: np.ndarray) -> np.ndarray:
        """
        Convert backbone features to 256-dim embedding.

        Args:
            features: Feature map of shape (N, C, H, W)

        Returns:
            Embeddings of shape (N, 256)
        """
        # Global Average Pooling: (N, C, H, W) -> (N, C)
        pooled = features.mean(axis=(2, 3))

        # Project to target dimension: (N, C) @ (C, 256) -> (N, 256)
        projected = pooled @ self.projection

        # L2 normalize
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = projected / norms

        return normalized

    def extract_embedding(self, image_path: str) -> np.ndarray:
        """
        Extract 256-dimensional embedding from a single image.

        Args:
            image_path: Path to the image file

        Returns:
            numpy array of shape (256,)
        """
        img = self._preprocess(image_path)
        img_batch = img[np.newaxis, ...]  # (1, 3, H, W)

        outputs = self.session.run(
            [self.feature_node_name], {self.input_name: img_batch}
        )
        features = outputs[0]
        embedding = self._features_to_embedding(features)
        return embedding.squeeze().astype(np.float32)

    def extract_batch(self, image_paths: list) -> np.ndarray:
        """
        Extract embeddings from a batch of images.

        Args:
            image_paths: List of image file paths

        Returns:
            numpy array of shape (batch_size, 256)
        """
        images = []
        valid_indices = []
        for i, path in enumerate(image_paths):
            try:
                img = self._preprocess(path)
                images.append(img)
                valid_indices.append(i)
            except Exception as e:
                print(f"Error loading {path}: {e}")

        if not images:
            return np.array([])

        batch = np.stack(images, axis=0)  # (N, 3, H, W)

        outputs = self.session.run(
            [self.feature_node_name], {self.input_name: batch}
        )
        features = outputs[0]
        embeddings = self._features_to_embedding(features)
        return embeddings.astype(np.float32)


def extract_embeddings_from_directory(
    extractor: NudeNetEmbeddingExtractor,
    input_dir: str,
    output_dir: str,
    batch_size: int = 32,
):
    """
    Extract embeddings from all images in a directory and save as .npy files.

    Args:
        extractor: NudeNetEmbeddingExtractor instance
        input_dir: Directory containing images
        output_dir: Directory to save embeddings
        batch_size: Batch size for processing
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    image_files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ]

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images in {input_dir}")

    for i in tqdm(range(0, len(image_files), batch_size), desc=f"Processing {input_dir}"):
        batch_files = image_files[i:i + batch_size]
        batch_paths = [str(f) for f in batch_files]

        embeddings = extractor.extract_batch(batch_paths)

        for file_path, embedding in zip(batch_files, embeddings):
            output_file = output_path / f"{file_path.stem}.npy"
            np.save(output_file, embedding.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(
        description="Extract NudeNet backbone embeddings for pNSFWMedia training"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="dataset/images",
        help="Root directory containing sfw/ and nsfw/ subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/embeddings",
        help="Root directory to save embeddings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--projection-path",
        type=str,
        default="models/nudenet_projection.npy",
        help="Path to save/load projection matrix",
    )
    parser.add_argument(
        "--feature-node",
        type=str,
        default=None,
        help="ONNX node name to extract features from (default: auto-detect)",
    )
    parser.add_argument(
        "--list-nodes",
        action="store_true",
        help="List all ONNX nodes and exit (for debugging)",
    )

    args = parser.parse_args()

    # Debug mode: list ONNX nodes
    if args.list_nodes:
        model_path = _find_nudenet_model_path()
        _list_onnx_nodes(model_path)
        return

    # Initialize extractor
    extractor = NudeNetEmbeddingExtractor(
        output_dim=256,
        feature_node=args.feature_node,
        projection_path=args.projection_path,
    )

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    classes = ["sfw", "nsfw"]
    for cls in classes:
        input_dir = input_root / cls
        output_dir = output_root / cls

        if input_dir.exists():
            print(f"\nProcessing {cls}...")
            extract_embeddings_from_directory(
                extractor,
                str(input_dir),
                str(output_dir),
                batch_size=args.batch_size,
            )
        else:
            print(f"Skipping {input_dir} (not found)")

    print("\nEmbedding extraction complete!")
    print(f"Embeddings saved to: {output_root}")


if __name__ == "__main__":
    main()
