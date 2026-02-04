"""
Model architectures for Semantic Feature Migration (SFM).

Components:
  - PerturbationGenerator: U-Net encoder-decoder that produces image perturbations
  - FeatureDifferenceConditioner: Learns NSFW/SFW centroid gap and conditions generator
  - PerceptualSimilarityNet: Multi-scale VGG feature distance for human-vision-aware constraint
  - KerasClassifierBridge: Reconstructs the pNSFWMedia Keras classifier in PyTorch
  - DifferentiableCLIPPipeline: End-to-end differentiable CLIP + projection + classifier
"""

import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv2d -> GroupNorm -> LeakyReLU"""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """Two-conv residual block with optional channel adjustment."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.GroupNorm(min(32, channels), channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.GroupNorm(min(32, channels), channels),
        )

    def forward(self, x):
        return x + self.block(x)


# ---------------------------------------------------------------------------
# Perturbation Generator (U-Net with feature-difference conditioning)
# ---------------------------------------------------------------------------

class PerturbationGenerator(nn.Module):
    """
    Encoder-decoder network that produces a bounded perturbation delta
    for a given input image.

    Architecture
    ------------
    Encoder:  Conv blocks with stride-2 downsampling (4 levels)
    Bottleneck: Residual blocks + FiLM conditioning from feature difference
    Decoder:  Transposed-conv upsampling with skip connections (4 levels)
    Output:   tanh * epsilon  (pixel-space perturbation, bounded)

    The generator receives a *conditioning vector* (NSFW-SFW feature
    difference) that is injected at the bottleneck via FiLM (Feature-wise
    Linear Modulation).  This makes the perturbation aware of *which
    direction* in feature space the image needs to move.
    """

    def __init__(self, epsilon: float = 0.03, cond_dim: int = 256):
        """
        Args:
            epsilon: Maximum L-inf perturbation magnitude (pixel range [0,1]).
            cond_dim: Dimensionality of the conditioning vector (embedding dim).
        """
        super().__init__()
        self.epsilon = epsilon

        # --- Encoder ---
        self.enc1 = ConvBlock(3, 32, stride=1)       # H x W
        self.enc2 = ConvBlock(32, 64, stride=2)       # H/2
        self.enc3 = ConvBlock(64, 128, stride=2)      # H/4
        self.enc4 = ConvBlock(128, 256, stride=2)     # H/8

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
        )

        # FiLM conditioning: project cond_dim -> (gamma, beta) for 256 channels
        self.film_proj = nn.Sequential(
            nn.Linear(cond_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256 * 2),  # gamma + beta
        )

        # --- Decoder ---
        self.up4 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.dec4 = ConvBlock(256, 128)  # concat with enc3 skip

        self.up3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.dec3 = ConvBlock(128, 64)   # concat with enc2 skip

        self.up2 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False)
        self.dec2 = ConvBlock(64, 32)    # concat with enc1 skip

        # Output head: 32 -> 3 channels, tanh activation
        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor [B, 3, H, W] in [0, 1].
            cond: Conditioning vector [B, cond_dim].

        Returns:
            Perturbed image [B, 3, H, W] clamped to [0, 1].
        """
        # Encoder
        e1 = self.enc1(x)     # [B, 32, H, W]
        e2 = self.enc2(e1)    # [B, 64, H/2, W/2]
        e3 = self.enc3(e2)    # [B, 128, H/4, W/4]
        e4 = self.enc4(e3)    # [B, 256, H/8, W/8]

        # Bottleneck + FiLM
        b = self.bottleneck(e4)
        film_params = self.film_proj(cond)                # [B, 512]
        gamma, beta = film_params.chunk(2, dim=1)         # each [B, 256]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1) + 1.0   # center around 1
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        b = gamma * b + beta

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = _match_size(d4, e3)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))

        d3 = self.up3(d4)
        d3 = _match_size(d3, e2)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = self.up2(d3)
        d2 = _match_size(d2, e1)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        # Output perturbation
        delta = self.out_conv(d2) * self.epsilon  # bounded to [-eps, +eps]

        perturbed = torch.clamp(x + delta, 0.0, 1.0)
        return perturbed


def _match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Crop or pad x to match spatial size of ref."""
    if x.shape[2:] != ref.shape[2:]:
        x = F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)
    return x


# ---------------------------------------------------------------------------
# Feature Difference Conditioner
# ---------------------------------------------------------------------------

class FeatureDifferenceConditioner(nn.Module):
    """
    Maintains exponential-moving-average centroids for SFW and NSFW
    embeddings and computes a conditioning vector that encodes the
    direction from NSFW to SFW in feature space.

    During training the centroids are updated with each batch.
    The conditioning vector fed to the generator is:
        cond = MLP(sfw_centroid - nsfw_embedding)
    so the generator knows *how far* and *in which direction* the
    current NSFW sample needs to move.
    """

    def __init__(self, embed_dim: int = 256, momentum: float = 0.99):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("sfw_centroid", torch.zeros(embed_dim))
        self.register_buffer("nsfw_centroid", torch.zeros(embed_dim))
        self.register_buffer("initialized", torch.tensor(False))

        # Small MLP to transform the raw difference into a richer signal
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    @torch.no_grad()
    def update_centroids(
        self, sfw_embeds: torch.Tensor, nsfw_embeds: torch.Tensor
    ):
        """Update running centroids with current batch embeddings."""
        sfw_mean = sfw_embeds.mean(dim=0)
        nsfw_mean = nsfw_embeds.mean(dim=0)

        if not self.initialized:
            self.sfw_centroid.copy_(sfw_mean)
            self.nsfw_centroid.copy_(nsfw_mean)
            self.initialized.fill_(True)
        else:
            m = self.momentum
            self.sfw_centroid.mul_(m).add_(sfw_mean, alpha=1 - m)
            self.nsfw_centroid.mul_(m).add_(nsfw_mean, alpha=1 - m)

    def forward(self, nsfw_embeds: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample conditioning vector.

        Args:
            nsfw_embeds: [B, embed_dim] embeddings of NSFW images.

        Returns:
            cond: [B, embed_dim] conditioning vectors.
        """
        # Direction from each NSFW sample toward the SFW centroid
        diff = self.sfw_centroid.unsqueeze(0) - nsfw_embeds  # [B, D]
        return self.proj(diff)


# ---------------------------------------------------------------------------
# Perceptual Similarity Network
# ---------------------------------------------------------------------------

class PerceptualSimilarityNet(nn.Module):
    """
    Multi-scale perceptual similarity using VGG-16 features.

    Extracts features at relu1_2, relu2_2, relu3_3, relu4_3 and computes
    channel-normalized L2 distances at each scale, weighted and summed.

    This approximates human visual perception: low-level texture, mid-level
    patterns, and high-level structure are all compared.
    """

    # Feature extraction points in VGG-16
    LAYER_NAMES = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
    LAYER_INDICES = [3, 8, 15, 22]  # Indices in vgg16.features

    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
        features = vgg.features

        # Split VGG into slices up to each extraction point
        slices = []
        prev = 0
        for idx in self.LAYER_INDICES:
            slices.append(nn.Sequential(*list(features.children())[prev:idx + 1]))
            prev = idx + 1
        self.slices = nn.ModuleList(slices)

        # Learnable per-layer weights (initialized uniformly)
        self.layer_weights = nn.Parameter(
            torch.ones(len(self.LAYER_INDICES)) / len(self.LAYER_INDICES)
        )

        # Freeze VGG weights
        for p in self.slices.parameters():
            p.requires_grad = False

        # ImageNet normalization constants
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def _extract_features(self, x: torch.Tensor) -> list:
        x = self._normalize(x)
        feats = []
        for s in self.slices:
            x = s(x)
            feats.append(x)
        return feats

    def forward(
        self, img1: torch.Tensor, img2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute perceptual distance between two images.

        Args:
            img1, img2: [B, 3, H, W] in [0, 1].

        Returns:
            Scalar perceptual distance (mean over batch).
        """
        feats1 = self._extract_features(img1)
        feats2 = self._extract_features(img2)

        # Softmax over layer weights so they sum to 1
        weights = F.softmax(self.layer_weights, dim=0)

        loss = 0.0
        for w, f1, f2 in zip(weights, feats1, feats2):
            # Channel-normalize then compute L2
            f1_norm = f1 / (f1.norm(dim=1, keepdim=True) + 1e-8)
            f2_norm = f2 / (f2.norm(dim=1, keepdim=True) + 1e-8)
            loss = loss + w * (f1_norm - f2_norm).pow(2).mean()

        return loss


# ---------------------------------------------------------------------------
# Keras Classifier -> PyTorch Bridge
# ---------------------------------------------------------------------------

class KerasClassifierBridge(nn.Module):
    """
    Reconstructs the pNSFWMedia Keras classifier in pure PyTorch
    so that the full pipeline is differentiable.

    Architecture (matching train_classifier.py default config):
        BatchNorm1d(256)
        Linear(256, units) + activation
        [optional: BatchNorm1d + Linear + activation]
        Linear(units, 1) + Sigmoid

    Weights can be loaded from a trained .keras file.
    """

    def __init__(
        self,
        input_dim: int = 256,
        units: int = 256,
        num_layers: int = 1,
        activation: str = "tanh",
    ):
        super().__init__()
        layers = []

        for i in range(num_layers):
            in_features = input_dim if i == 0 else units
            layers.append(nn.BatchNorm1d(in_features))
            layers.append(nn.Linear(in_features, units))
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "gelu":
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU())

        layers.append(nn.Linear(units, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 256] embedding.
        Returns:
            [B, 1] NSFW probability.
        """
        return self.net(x)

    @classmethod
    def from_keras(cls, keras_path: str, device: str = "cpu"):
        """
        Load weights from a trained .keras / .h5 model file.

        This inspects the Keras model to determine architecture parameters
        and copies the weights into the PyTorch equivalent.
        """
        import tensorflow as tf

        tf_model = tf.keras.models.load_model(keras_path)
        tf_model.summary()

        # Infer architecture from Keras layers
        bn_layers = [
            l for l in tf_model.layers if isinstance(l, tf.keras.layers.BatchNormalization)
        ]
        dense_layers = [
            l for l in tf_model.layers if isinstance(l, tf.keras.layers.Dense)
        ]

        # The last Dense is output (1 unit, sigmoid); the rest are hidden
        hidden_dense = dense_layers[:-1]
        output_dense = dense_layers[-1]

        num_layers = len(hidden_dense)
        units = hidden_dense[0].units if hidden_dense else 256

        # Detect activation
        activation = "tanh"
        if hidden_dense:
            act_config = hidden_dense[0].get_config().get("activation", "tanh")
            if isinstance(act_config, dict):
                activation = act_config.get("class_name", "tanh").lower()
            else:
                activation = act_config

        print(f"[KerasClassifierBridge] Detected: num_layers={num_layers}, "
              f"units={units}, activation={activation}")

        # Create PyTorch model
        model = cls(
            input_dim=256, units=units,
            num_layers=num_layers, activation=activation,
        )

        # Copy weights
        pt_idx = 0
        tf_bn_idx = 0
        tf_dense_idx = 0

        for module in model.net:
            if isinstance(module, nn.BatchNorm1d) and tf_bn_idx < len(bn_layers):
                tf_bn = bn_layers[tf_bn_idx]
                w = tf_bn.get_weights()  # [gamma, beta, moving_mean, moving_var]
                module.weight.data = torch.tensor(w[0], dtype=torch.float32)
                module.bias.data = torch.tensor(w[1], dtype=torch.float32)
                module.running_mean.data = torch.tensor(w[2], dtype=torch.float32)
                module.running_var.data = torch.tensor(w[3], dtype=torch.float32)
                tf_bn_idx += 1

            elif isinstance(module, nn.Linear):
                if tf_dense_idx < len(dense_layers):
                    tf_dense = dense_layers[tf_dense_idx]
                    w = tf_dense.get_weights()
                    # Keras Dense weight shape: (in, out), PyTorch: (out, in)
                    module.weight.data = torch.tensor(
                        w[0].T, dtype=torch.float32
                    )
                    if len(w) > 1:
                        module.bias.data = torch.tensor(
                            w[1], dtype=torch.float32
                        )
                    tf_dense_idx += 1

        model = model.to(device)
        model.eval()
        print(f"[KerasClassifierBridge] Weights loaded from {keras_path}")
        return model


# ---------------------------------------------------------------------------
# Differentiable CLIP Pipeline
# ---------------------------------------------------------------------------

class DifferentiableCLIPPipeline(nn.Module):
    """
    End-to-end differentiable pipeline:
        Image -> CLIP encoder -> Linear projection -> L2 norm -> 256-dim embedding

    The CLIP backbone is frozen; only the projection layer uses pre-trained
    weights from the pNSFWMedia project.
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        projection_path: str = None,
        output_dim: int = 256,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.output_dim = output_dim

        try:
            import clip as clip_module
        except ImportError:
            raise ImportError(
                "CLIP is required: pip install git+https://github.com/openai/CLIP.git"
            )

        self.clip_model, self.clip_preprocess = clip_module.load(
            clip_model_name, device=device
        )
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # Determine CLIP output dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            clip_dim = self.clip_model.encode_image(dummy).shape[-1]

        self.projection = nn.Linear(clip_dim, output_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)

        if projection_path and os.path.exists(projection_path):
            state = torch.load(projection_path, map_location=device)
            self.projection.load_state_dict(state)
            print(f"[CLIPPipeline] Loaded projection from {projection_path}")

        # Freeze projection as well (we don't want to modify the extractor)
        for p in self.projection.parameters():
            p.requires_grad = False

        # CLIP normalization (applied to images already in [0,1])
        self.register_buffer(
            "clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
        )

    def preprocess_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize [0,1] image tensor for CLIP input.
        Resizes to 224x224 and applies CLIP normalization.
        """
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        x = (x - self.clip_mean) / self.clip_std
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] images in [0, 1].
        Returns:
            [B, 256] L2-normalized embeddings.
        """
        x = self.preprocess_tensor(x)
        features = self.clip_model.encode_image(x).float()
        projected = self.projection(features)
        projected = F.normalize(projected, p=2, dim=-1)
        return projected
