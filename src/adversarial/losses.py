"""
Loss functions for Semantic Feature Migration (SFM) training.

Composite loss:
  L_total = λ_cls  * L_classification
          + λ_feat * L_feature_migration
          + λ_perc * L_perceptual
          + λ_mag  * L_magnitude

Design rationale
----------------
- L_classification: Forces the classifier to output low NSFW probability
  for perturbed images. Uses soft-target BCE with label 0.0 (SFW).

- L_feature_migration: Pulls the perturbed embedding toward the SFW
  centroid in L2 space. This is the *semantic misdirection* component
  that distinguishes this approach from a naive confidence reduction.

- L_perceptual: Multi-scale VGG feature distance between original and
  perturbed images. Ensures visual fidelity to human observers.

- L_magnitude: Mean L2 norm of the pixel-space perturbation. Acts as
  a regularizer to keep perturbations small.

None of these losses implement FGSM, PGD, C&W, or DeepFool.
The generator is trained via standard backpropagation through the
frozen pipeline (CLIP + classifier).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticFeatureMigrationLoss(nn.Module):
    """
    Combined loss for adversarial perturbation training.

    All sub-losses are differentiable with respect to the generator
    parameters. The CLIP pipeline and classifier are frozen.
    """

    def __init__(
        self,
        perceptual_net,
        lambda_cls: float = 1.0,
        lambda_feat: float = 2.0,
        lambda_perc: float = 1.0,
        lambda_mag: float = 10.0,
        sfw_target: float = 0.05,
    ):
        """
        Args:
            perceptual_net: PerceptualSimilarityNet instance.
            lambda_cls: Weight for classification loss.
            lambda_feat: Weight for feature migration loss.
            lambda_perc: Weight for perceptual similarity loss.
            lambda_mag: Weight for perturbation magnitude loss.
            sfw_target: Target probability for the classifier output
                        (close to 0 = confident SFW).
        """
        super().__init__()
        self.perceptual_net = perceptual_net
        self.lambda_cls = lambda_cls
        self.lambda_feat = lambda_feat
        self.lambda_perc = lambda_perc
        self.lambda_mag = lambda_mag
        self.sfw_target = sfw_target

    def classification_loss(
        self, pred_prob: torch.Tensor
    ) -> torch.Tensor:
        """
        Soft-target BCE pushing predictions toward SFW (0.0).

        Args:
            pred_prob: [B, 1] classifier output probabilities.
        Returns:
            Scalar loss.
        """
        target = torch.full_like(pred_prob, self.sfw_target)
        return F.binary_cross_entropy(pred_prob, target)

    def feature_migration_loss(
        self,
        perturbed_embed: torch.Tensor,
        sfw_centroid: torch.Tensor,
        sfw_batch_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pull perturbed NSFW embeddings toward SFW distribution.

        Two components:
          1. Distance to running SFW centroid (global direction)
          2. Cosine similarity to batch-level SFW mean (local alignment)

        Args:
            perturbed_embed: [B, D] perturbed NSFW embeddings.
            sfw_centroid: [D] running SFW centroid.
            sfw_batch_embed: [B, D] SFW embeddings from current batch.
        Returns:
            Scalar loss.
        """
        # Global: L2 distance to SFW centroid
        centroid_dist = F.mse_loss(
            perturbed_embed, sfw_centroid.unsqueeze(0).expand_as(perturbed_embed)
        )

        # Local: maximize cosine similarity to batch SFW mean
        sfw_mean = sfw_batch_embed.mean(dim=0, keepdim=True)
        cos_sim = F.cosine_similarity(perturbed_embed, sfw_mean.expand_as(perturbed_embed), dim=1)
        cos_loss = 1.0 - cos_sim.mean()  # minimize (1 - cos_sim)

        return centroid_dist + 0.5 * cos_loss

    def perceptual_loss(
        self,
        original: torch.Tensor,
        perturbed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Multi-scale perceptual distance (VGG-based).

        Args:
            original: [B, 3, H, W] original images.
            perturbed: [B, 3, H, W] perturbed images.
        Returns:
            Scalar loss.
        """
        return self.perceptual_net(original, perturbed)

    def magnitude_loss(
        self,
        original: torch.Tensor,
        perturbed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean-squared perturbation magnitude in pixel space.

        Args:
            original: [B, 3, H, W] original images.
            perturbed: [B, 3, H, W] perturbed images.
        Returns:
            Scalar loss.
        """
        delta = perturbed - original
        return delta.pow(2).mean()

    def forward(
        self,
        original_images: torch.Tensor,
        perturbed_images: torch.Tensor,
        perturbed_embed: torch.Tensor,
        classifier_pred: torch.Tensor,
        sfw_centroid: torch.Tensor,
        sfw_batch_embed: torch.Tensor,
    ) -> dict:
        """
        Compute all sub-losses and the weighted total.

        Args:
            original_images: [B, 3, H, W] clean NSFW images.
            perturbed_images: [B, 3, H, W] perturbed images.
            perturbed_embed: [B, D] embeddings of perturbed images.
            classifier_pred: [B, 1] classifier output for perturbed.
            sfw_centroid: [D] SFW centroid from conditioner.
            sfw_batch_embed: [B, D] batch SFW embeddings.

        Returns:
            Dict with keys: total, cls, feat, perc, mag (all scalar tensors).
        """
        l_cls = self.classification_loss(classifier_pred)
        l_feat = self.feature_migration_loss(
            perturbed_embed, sfw_centroid, sfw_batch_embed
        )
        l_perc = self.perceptual_loss(original_images, perturbed_images)
        l_mag = self.magnitude_loss(original_images, perturbed_images)

        total = (
            self.lambda_cls * l_cls
            + self.lambda_feat * l_feat
            + self.lambda_perc * l_perc
            + self.lambda_mag * l_mag
        )

        return {
            "total": total,
            "cls": l_cls.detach(),
            "feat": l_feat.detach(),
            "perc": l_perc.detach(),
            "mag": l_mag.detach(),
        }
