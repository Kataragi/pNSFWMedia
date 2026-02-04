"""
Adversarial Perturbation Module for pNSFWMedia

Semantic Feature Migration (SFM) approach:
A learned perturbation generator that shifts NSFW image embeddings
toward the SFW feature distribution, causing misclassification.

This is NOT an iterative gradient attack (FGSM/PGD) nor an optimization-based
attack (C&W/DeepFool). It is a feed-forward generative model that learns
the NSFW-to-SFW feature migration in a single forward pass.
"""

__version__ = "1.0.0"
