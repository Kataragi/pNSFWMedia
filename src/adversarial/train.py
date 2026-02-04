#!/usr/bin/env python3
"""
Training script for Semantic Feature Migration (SFM) adversarial perturbation.

This trains a PerturbationGenerator that learns to add minimal, perceptually
invisible noise to NSFW images so the pNSFWMedia classifier misclassifies
them as SFW.

Approach
--------
1. Load paired NSFW/SFW images.
2. NSFW images are forwarded through the frozen CLIP pipeline to get
   their embeddings.  SFW images likewise to establish the target
   distribution.
3. The FeatureDifferenceConditioner computes a per-sample conditioning
   vector (direction from NSFW toward SFW in feature space).
4. The PerturbationGenerator produces a perturbation conditioned on
   this vector and adds it to the original NSFW image.
5. The perturbed image passes through the frozen CLIP pipeline and
   classifier.
6. Four losses jointly update the generator:
     - Classification loss (push toward SFW label)
     - Feature migration loss (move embedding toward SFW centroid)
     - Perceptual similarity loss (keep visual fidelity)
     - Magnitude loss (minimize perturbation size)

This is NOT FGSM, PGD, C&W, or DeepFool. It is a learned feed-forward
generator trained via standard back-propagation.

CUDA Compatibility
------------------
If CUDA errors occur, set the following environment variables BEFORE
running this script:

    CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python src/adversarial/train.py ...

This forces synchronous execution so that CUDA errors are reported at
the exact line of failure.

Usage
-----
    python src/adversarial/train.py \\
        --image-dir dataset/images \\
        --classifier-path models/pnsfwmedia_classifier.keras \\
        --projection-path models/clip_projection.pt \\
        --epochs 50 \\
        --batch-size 8 \\
        --lr 2e-4 \\
        --epsilon 0.03 \\
        --output-dir models/adversarial

Alternatively, with CUDA debug mode:

    CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \\
    python src/adversarial/train.py \\
        --image-dir dataset/images \\
        --classifier-path models/pnsfwmedia_classifier.keras \\
        --epochs 50
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adversarial.models import (
    DifferentiableCLIPPipeline,
    FeatureDifferenceConditioner,
    KerasClassifierBridge,
    PerceptualSimilarityNet,
    PerturbationGenerator,
)
from adversarial.losses import SemanticFeatureMigrationLoss
from adversarial.dataset import create_dataloader


# ---------------------------------------------------------------------------
# CUDA setup & diagnostics
# ---------------------------------------------------------------------------

def setup_device(force_cpu: bool = False) -> torch.device:
    """
    Configure PyTorch device with CUDA diagnostics.

    Respects CUDA_VISIBLE_DEVICES and CUDA_LAUNCH_BLOCKING env vars.
    """
    if force_cpu or not torch.cuda.is_available():
        print("[Device] Using CPU")
        return torch.device("cpu")

    # Report CUDA debug env vars
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")
    clb = os.environ.get("CUDA_LAUNCH_BLOCKING", "(not set)")
    print(f"[Device] CUDA_VISIBLE_DEVICES = {cvd}")
    print(f"[Device] CUDA_LAUNCH_BLOCKING = {clb}")

    device = torch.device("cuda")
    print(f"[Device] Using CUDA: {torch.cuda.get_device_name(0)}")
    print(f"[Device] CUDA version: {torch.version.cuda}")
    print(f"[Device] cuDNN version: {torch.backends.cudnn.version()}")
    print(f"[Device] GPU memory: "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Enable TF32 for Ampere+ GPUs (faster, negligible precision loss)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    return device


# ---------------------------------------------------------------------------
# Validation utilities
# ---------------------------------------------------------------------------

def validate_classifier_load(classifier_path: str, device: torch.device):
    """
    Pre-validate that the Keras classifier can be loaded and converted
    without shape mismatches or undefined-layer errors.
    """
    print(f"\n{'='*60}")
    print("Validating classifier model: {classifier_path}")
    print(f"{'='*60}")

    if not os.path.exists(classifier_path):
        raise FileNotFoundError(
            f"Classifier model not found: {classifier_path}\n"
            f"Train one first with: python src/train_classifier.py"
        )

    # Attempt load
    classifier = KerasClassifierBridge.from_keras(
        classifier_path, device=str(device)
    )

    # Shape validation: feed a dummy 256-dim embedding
    dummy = torch.randn(2, 256, device=device)
    with torch.no_grad():
        out = classifier(dummy)

    assert out.shape == (2, 1), (
        f"Classifier output shape mismatch: expected (2, 1), got {out.shape}"
    )
    assert 0.0 <= out.min() and out.max() <= 1.0, (
        f"Classifier output out of [0,1] range: min={out.min()}, max={out.max()}"
    )

    print("[Validate] Classifier loaded and verified successfully")
    print(f"[Validate] Test output: {out.squeeze().tolist()}")
    return classifier


def validate_clip_pipeline(
    clip_model: str, projection_path: str, device: torch.device
):
    """Pre-validate CLIP pipeline with dummy input."""
    print(f"\n{'='*60}")
    print("Validating CLIP pipeline")
    print(f"{'='*60}")

    pipeline = DifferentiableCLIPPipeline(
        clip_model_name=clip_model,
        projection_path=projection_path,
        device=str(device),
    )
    pipeline = pipeline.to(device)

    dummy = torch.randn(2, 3, 224, 224, device=device).clamp(0, 1)
    with torch.no_grad():
        emb = pipeline(dummy)

    assert emb.shape == (2, 256), (
        f"Pipeline output shape mismatch: expected (2, 256), got {emb.shape}"
    )
    norms = emb.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), (
        f"Embeddings not L2-normalized: norms={norms.tolist()}"
    )

    print("[Validate] CLIP pipeline verified successfully")
    return pipeline


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    epoch: int,
    dataloader,
    generator: PerturbationGenerator,
    conditioner: FeatureDifferenceConditioner,
    clip_pipeline: DifferentiableCLIPPipeline,
    classifier: KerasClassifierBridge,
    criterion: SemanticFeatureMigrationLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    writer: SummaryWriter,
    global_step: int,
) -> tuple:
    """
    Run one training epoch.

    Returns:
        (avg_loss_dict, global_step)
    """
    generator.train()
    conditioner.train()

    running = {"total": 0.0, "cls": 0.0, "feat": 0.0, "perc": 0.0, "mag": 0.0}
    n_batches = 0
    attack_success = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)

    for batch in pbar:
        nsfw_images = batch["nsfw_image"].to(device)
        sfw_images = batch["sfw_image"].to(device)
        B = nsfw_images.size(0)

        # --- Forward through frozen pipeline ---
        with torch.no_grad():
            nsfw_embeds = clip_pipeline(nsfw_images)   # [B, 256]
            sfw_embeds = clip_pipeline(sfw_images)     # [B, 256]

        # Update running centroids
        conditioner.update_centroids(sfw_embeds, nsfw_embeds)

        # Conditioning vector for generator
        cond = conditioner(nsfw_embeds)   # [B, 256]

        # --- Generate perturbation ---
        perturbed_images = generator(nsfw_images, cond)  # [B, 3, H, W]

        # --- Forward perturbed through pipeline ---
        perturbed_embeds = clip_pipeline(perturbed_images)  # [B, 256]
        classifier_pred = classifier(perturbed_embeds)      # [B, 1]

        # --- Compute loss ---
        losses = criterion(
            original_images=nsfw_images,
            perturbed_images=perturbed_images,
            perturbed_embed=perturbed_embeds,
            classifier_pred=classifier_pred,
            sfw_centroid=conditioner.sfw_centroid,
            sfw_batch_embed=sfw_embeds,
        )

        # --- Backward ---
        optimizer.zero_grad()
        losses["total"].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(conditioner.parameters(), max_norm=1.0)

        optimizer.step()

        # --- Logging ---
        for key in running:
            running[key] += losses[key].item()
        n_batches += 1

        # Attack success rate: classifier output < 0.5 means SFW
        with torch.no_grad():
            attack_success += (classifier_pred < 0.5).sum().item()
            total_samples += B

        asr = attack_success / total_samples if total_samples > 0 else 0.0
        pbar.set_postfix({
            "loss": f"{losses['total'].item():.4f}",
            "cls": f"{losses['cls'].item():.4f}",
            "ASR": f"{asr:.1%}",
        })

        # TensorBoard
        if writer is not None:
            writer.add_scalar("train/loss_total", losses["total"].item(), global_step)
            writer.add_scalar("train/loss_cls", losses["cls"].item(), global_step)
            writer.add_scalar("train/loss_feat", losses["feat"].item(), global_step)
            writer.add_scalar("train/loss_perc", losses["perc"].item(), global_step)
            writer.add_scalar("train/loss_mag", losses["mag"].item(), global_step)
            writer.add_scalar("train/attack_success_rate", asr, global_step)

        global_step += 1

    # Averages
    avg = {k: v / max(n_batches, 1) for k, v in running.items()}
    avg["attack_success_rate"] = attack_success / max(total_samples, 1)
    return avg, global_step


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    dataloader,
    generator: PerturbationGenerator,
    conditioner: FeatureDifferenceConditioner,
    clip_pipeline: DifferentiableCLIPPipeline,
    classifier: KerasClassifierBridge,
    criterion: SemanticFeatureMigrationLoss,
    device: torch.device,
) -> dict:
    """Evaluate on held-out data."""
    generator.eval()

    running = {"total": 0.0, "cls": 0.0, "feat": 0.0, "perc": 0.0, "mag": 0.0}
    n_batches = 0
    attack_success = 0
    total_samples = 0
    all_probs_original = []
    all_probs_perturbed = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        nsfw_images = batch["nsfw_image"].to(device)
        sfw_images = batch["sfw_image"].to(device)
        B = nsfw_images.size(0)

        nsfw_embeds = clip_pipeline(nsfw_images)
        sfw_embeds = clip_pipeline(sfw_images)

        cond = conditioner(nsfw_embeds)
        perturbed_images = generator(nsfw_images, cond)
        perturbed_embeds = clip_pipeline(perturbed_images)
        classifier_pred = classifier(perturbed_embeds)

        # Original predictions for comparison
        original_pred = classifier(nsfw_embeds)
        all_probs_original.extend(original_pred.squeeze().cpu().tolist())
        all_probs_perturbed.extend(classifier_pred.squeeze().cpu().tolist())

        losses = criterion(
            original_images=nsfw_images,
            perturbed_images=perturbed_images,
            perturbed_embed=perturbed_embeds,
            classifier_pred=classifier_pred,
            sfw_centroid=conditioner.sfw_centroid,
            sfw_batch_embed=sfw_embeds,
        )

        for key in running:
            running[key] += losses[key].item()
        n_batches += 1
        attack_success += (classifier_pred < 0.5).sum().item()
        total_samples += B

    avg = {k: v / max(n_batches, 1) for k, v in running.items()}
    avg["attack_success_rate"] = attack_success / max(total_samples, 1)
    avg["mean_original_prob"] = np.mean(all_probs_original) if all_probs_original else 0.0
    avg["mean_perturbed_prob"] = np.mean(all_probs_perturbed) if all_probs_perturbed else 0.0

    return avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train SFM adversarial perturbation generator for pNSFWMedia"
    )
    # Paths
    parser.add_argument(
        "--image-dir", type=str, default="dataset/images",
        help="Directory with nsfw/ and sfw/ image subdirectories",
    )
    parser.add_argument(
        "--classifier-path", type=str,
        default="models/pnsfwmedia_classifier.keras",
        help="Path to trained pNSFWMedia Keras classifier",
    )
    parser.add_argument(
        "--projection-path", type=str,
        default="models/clip_projection.pt",
        help="Path to CLIP projection layer weights",
    )
    parser.add_argument(
        "--output-dir", type=str, default="models/adversarial",
        help="Directory to save trained generator checkpoints",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs/adversarial",
        help="TensorBoard log directory",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Perturbation control
    parser.add_argument(
        "--epsilon", type=float, default=0.03,
        help="Max L-inf perturbation magnitude (pixel range [0,1])",
    )

    # Loss weights
    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--lambda-feat", type=float, default=2.0)
    parser.add_argument("--lambda-perc", type=float, default=1.0)
    parser.add_argument("--lambda-mag", type=float, default=10.0)
    parser.add_argument(
        "--sfw-target", type=float, default=0.05,
        help="Target NSFW probability for perturbed images (0.0 = confident SFW)",
    )

    # CLIP
    parser.add_argument(
        "--clip-model", type=str, default="ViT-B/32",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"],
    )

    # Device
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")

    # Checkpointing
    parser.add_argument(
        "--save-every", type=int, default=5,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = setup_device(force_cpu=args.cpu)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = f"{args.log_dir}/{timestamp}"
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=log_path)

    # Save config
    config = vars(args)
    config["timestamp"] = timestamp
    config["device"] = str(device)
    with open(f"{args.output_dir}/train_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print("Semantic Feature Migration - Adversarial Training")
    print(f"{'='*60}")
    print(f"Config: {json.dumps(config, indent=2)}")

    # -----------------------------------------------------------------------
    # Load frozen components
    # -----------------------------------------------------------------------
    classifier = validate_classifier_load(args.classifier_path, device)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    clip_pipeline = validate_clip_pipeline(
        args.clip_model, args.projection_path, device
    )
    clip_pipeline.eval()

    # -----------------------------------------------------------------------
    # Trainable components
    # -----------------------------------------------------------------------
    generator = PerturbationGenerator(
        epsilon=args.epsilon, cond_dim=256
    ).to(device)

    conditioner = FeatureDifferenceConditioner(
        embed_dim=256, momentum=0.99
    ).to(device)

    perceptual_net = PerceptualSimilarityNet().to(device)
    perceptual_net.eval()  # VGG backbone is frozen; only layer_weights train

    criterion = SemanticFeatureMigrationLoss(
        perceptual_net=perceptual_net,
        lambda_cls=args.lambda_cls,
        lambda_feat=args.lambda_feat,
        lambda_perc=args.lambda_perc,
        lambda_mag=args.lambda_mag,
        sfw_target=args.sfw_target,
    ).to(device)

    # Optimizer: generator params + conditioner MLP + perceptual layer weights
    trainable_params = (
        list(generator.parameters())
        + list(conditioner.proj.parameters())
        + [perceptual_net.layer_weights]
    )
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Loading dataset")
    print(f"{'='*60}")

    dataloader = create_dataloader(
        root_dir=args.image_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        augment=True,
        seed=args.seed,
    )

    # -----------------------------------------------------------------------
    # Resume from checkpoint
    # -----------------------------------------------------------------------
    start_epoch = 0
    global_step = 0

    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        generator.load_state_dict(ckpt["generator"])
        conditioner.load_state_dict(ckpt["conditioner"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        print(f"Resumed at epoch {start_epoch}, step {global_step}")

    # -----------------------------------------------------------------------
    # Parameter counts
    # -----------------------------------------------------------------------
    gen_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    cond_params = sum(p.numel() for p in conditioner.proj.parameters() if p.requires_grad)
    print(f"\n[Params] Generator: {gen_params:,}")
    print(f"[Params] Conditioner MLP: {cond_params:,}")
    print(f"[Params] Total trainable: {gen_params + cond_params + 4:,}")

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    best_asr = 0.0

    print(f"\n{'='*60}")
    print(f"Starting training: {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        avg_losses, global_step = train_one_epoch(
            epoch=epoch,
            dataloader=dataloader,
            generator=generator,
            conditioner=conditioner,
            clip_pipeline=clip_pipeline,
            classifier=classifier,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            writer=writer,
            global_step=global_step,
        )

        scheduler.step()
        elapsed = time.time() - t0

        # Log epoch summary
        asr = avg_losses["attack_success_rate"]
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"\n[Epoch {epoch:3d}] "
            f"loss={avg_losses['total']:.4f}  "
            f"cls={avg_losses['cls']:.4f}  "
            f"feat={avg_losses['feat']:.4f}  "
            f"perc={avg_losses['perc']:.4f}  "
            f"mag={avg_losses['mag']:.4f}  "
            f"ASR={asr:.1%}  "
            f"lr={lr_now:.2e}  "
            f"time={elapsed:.1f}s"
        )

        writer.add_scalar("epoch/loss", avg_losses["total"], epoch)
        writer.add_scalar("epoch/asr", asr, epoch)
        writer.add_scalar("epoch/lr", lr_now, epoch)

        # Save best
        if asr > best_asr:
            best_asr = asr
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "generator": generator.state_dict(),
                "conditioner": conditioner.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_asr": best_asr,
                "config": config,
            }, f"{args.output_dir}/sfm_best.pt")
            print(f"  -> New best ASR: {best_asr:.1%} (saved)")

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "generator": generator.state_dict(),
                "conditioner": conditioner.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_asr": best_asr,
                "config": config,
            }, f"{args.output_dir}/sfm_epoch{epoch:03d}.pt")

    # Save final
    torch.save({
        "epoch": args.epochs - 1,
        "global_step": global_step,
        "generator": generator.state_dict(),
        "conditioner": conditioner.state_dict(),
        "config": config,
        "best_asr": best_asr,
    }, f"{args.output_dir}/sfm_final.pt")

    writer.close()

    print(f"\n{'='*60}")
    print("Training Complete")
    print(f"{'='*60}")
    print(f"Best Attack Success Rate: {best_asr:.1%}")
    print(f"Checkpoints saved to: {args.output_dir}/")
    print(f"TensorBoard logs: {log_path}/")
    print(f"\nTo visualize:")
    print(f"  tensorboard --logdir={args.log_dir}")
    print(f"\nTo apply perturbations:")
    print(f"  python src/adversarial/apply.py --checkpoint {args.output_dir}/sfm_best.pt --image <path>")


if __name__ == "__main__":
    main()
