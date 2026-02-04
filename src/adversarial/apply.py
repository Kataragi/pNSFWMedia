#!/usr/bin/env python3
"""
SFM Adversarial Perturbation — Inference (Apply) Script

Applies a trained PerturbationGenerator to input images and outputs
perturbed images at their original resolution. Also displays the
pNSFWMedia classifier's judgment before and after perturbation.

Usage
-----
    # Single image
    python src/adversarial/apply.py \
        --checkpoint models/adversarial/sfm_final.pt \
        --classifier-path models/pnsfwmedia_classifier.keras \
        --projection-path models/clip_projection.pt \
        --image path/to/nsfw_image.jpg \
        --output-dir output/

    # Directory of images
    python src/adversarial/apply.py \
        --checkpoint models/adversarial/sfm_final.pt \
        --classifier-path models/pnsfwmedia_classifier.keras \
        --image-dir path/to/images/ \
        --output-dir output/

    # With CUDA debug
    CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \
    python src/adversarial/apply.py \
        --checkpoint models/adversarial/sfm_final.pt \
        --classifier-path models/pnsfwmedia_classifier.keras \
        --image path/to/image.jpg
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adversarial.models import (
    DifferentiableCLIPPipeline,
    FeatureDifferenceConditioner,
    KerasClassifierBridge,
    PerturbationGenerator,
)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ---------------------------------------------------------------------------
# Device setup (shared with train.py logic)
# ---------------------------------------------------------------------------

def setup_device(force_cpu: bool = False) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        print("[Device] Using CPU")
        return torch.device("cpu")

    device = torch.device("cuda")
    print(f"[Device] Using CUDA: {torch.cuda.get_device_name(0)}")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")
    clb = os.environ.get("CUDA_LAUNCH_BLOCKING", "(not set)")
    if cvd != "(not set)" or clb != "(not set)":
        print(f"[Device] CUDA_VISIBLE_DEVICES={cvd}  CUDA_LAUNCH_BLOCKING={clb}")
    return device


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def load_image_tensor(path: str, size: int = 224) -> tuple:
    """
    Load an image, return both the resized tensor and original PIL image.

    Returns:
        (tensor [1, 3, size, size] in [0,1],  original PIL.Image, original (W, H))
    """
    img = Image.open(path).convert("RGB")
    original_size = img.size  # (W, H)

    resized = img.resize((size, size), Image.BICUBIC)
    arr = np.array(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]

    return tensor, img, original_size


def save_perturbed_image(
    original_pil: Image.Image,
    perturbed_tensor_224: torch.Tensor,
    original_size: tuple,
    output_path: str,
):
    """
    Upscale the 224x224 perturbation delta to the original image resolution,
    apply it to the original image, and save.

    This preserves the full-resolution detail of the original while applying
    the learned perturbation pattern at the correct scale.

    Args:
        original_pil: Original PIL image at full resolution.
        perturbed_tensor_224: [1, 3, 224, 224] perturbed image tensor in [0,1].
        original_size: (W, H) of the original image.
        output_path: Where to save the result.
    """
    W, H = original_size

    # Convert original to tensor at 224x224 for delta computation
    orig_resized = original_pil.resize((224, 224), Image.BICUBIC)
    orig_arr = np.array(orig_resized, dtype=np.float32) / 255.0
    orig_tensor_224 = torch.from_numpy(orig_arr).permute(2, 0, 1).unsqueeze(0)

    # Compute delta at 224x224
    delta_224 = perturbed_tensor_224.cpu() - orig_tensor_224  # [1, 3, 224, 224]

    # Upscale delta to original resolution using bicubic interpolation
    delta_full = F.interpolate(
        delta_224,
        size=(H, W),
        mode="bicubic",
        align_corners=False,
    )  # [1, 3, H, W]

    # Apply delta to original full-res image
    orig_full_arr = np.array(original_pil, dtype=np.float32) / 255.0
    orig_full_tensor = torch.from_numpy(orig_full_arr).permute(2, 0, 1).unsqueeze(0)

    perturbed_full = torch.clamp(orig_full_tensor + delta_full, 0.0, 1.0)

    # Convert back to PIL and save
    result_arr = (perturbed_full.squeeze(0).permute(1, 2, 0).numpy() * 255.0)
    result_arr = result_arr.clip(0, 255).astype(np.uint8)
    result_img = Image.fromarray(result_arr)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Preserve original format where possible
    ext = Path(output_path).suffix.lower()
    save_kwargs = {}
    if ext in (".jpg", ".jpeg"):
        save_kwargs["quality"] = 95
    elif ext == ".png":
        save_kwargs["compress_level"] = 6
    elif ext == ".webp":
        save_kwargs["quality"] = 95

    result_img.save(output_path, **save_kwargs)
    return result_img


# ---------------------------------------------------------------------------
# Classification display
# ---------------------------------------------------------------------------

def format_classification(prob: float, threshold: float = 0.5) -> str:
    label = "NSFW" if prob >= threshold else "SFW"
    return f"{prob:.4f} ({label})"


def print_comparison(
    image_name: str,
    prob_before: float,
    prob_after: float,
    threshold: float = 0.5,
):
    """Print a formatted before/after comparison."""
    label_before = "NSFW" if prob_before >= threshold else "SFW"
    label_after = "NSFW" if prob_after >= threshold else "SFW"
    delta = prob_after - prob_before
    success = "SUCCESS" if (prob_before >= threshold and prob_after < threshold) else "---"

    print(f"  {image_name}")
    print(f"    Before : {prob_before:.4f}  [{label_before}]")
    print(f"    After  : {prob_after:.4f}  [{label_after}]")
    print(f"    Delta  : {delta:+.4f}")
    if success == "SUCCESS":
        print(f"    Result : ** NSFW -> SFW  (misclassification achieved) **")
    print()


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_perturbation(
    image_path: str,
    generator: PerturbationGenerator,
    conditioner: FeatureDifferenceConditioner,
    clip_pipeline: DifferentiableCLIPPipeline,
    classifier: KerasClassifierBridge,
    device: torch.device,
    image_size: int = 224,
) -> dict:
    """
    Apply the trained perturbation generator to a single image.

    Returns dict with probabilities and tensors.
    """
    tensor_224, original_pil, original_size = load_image_tensor(
        image_path, size=image_size
    )
    tensor_224 = tensor_224.to(device)

    # Embedding of original
    orig_embed = clip_pipeline(tensor_224)        # [1, 256]
    orig_pred = classifier(orig_embed)            # [1, 1]

    # Generate conditioning and perturbation
    cond = conditioner(orig_embed)                # [1, 256]
    perturbed_224 = generator(tensor_224, cond)   # [1, 3, 224, 224]

    # Classify perturbed
    pert_embed = clip_pipeline(perturbed_224)
    pert_pred = classifier(pert_embed)

    return {
        "original_pil": original_pil,
        "original_size": original_size,
        "perturbed_tensor_224": perturbed_224.cpu(),
        "prob_before": orig_pred.item(),
        "prob_after": pert_pred.item(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Apply SFM adversarial perturbation to images"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained SFM checkpoint (.pt)",
    )
    parser.add_argument(
        "--classifier-path", type=str,
        default="models/pnsfwmedia_classifier.keras",
        help="Path to pNSFWMedia Keras classifier",
    )
    parser.add_argument(
        "--projection-path", type=str,
        default="models/clip_projection.pt",
        help="Path to CLIP projection layer weights",
    )
    parser.add_argument(
        "--clip-model", type=str, default="ViT-B/32",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"],
    )

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Single image path")
    input_group.add_argument("--image-dir", type=str, help="Directory of images")

    # Output
    parser.add_argument(
        "--output-dir", type=str, default="output/adversarial",
        help="Directory to save perturbed images",
    )
    parser.add_argument(
        "--suffix", type=str, default="_perturbed",
        help="Suffix appended to output filenames (default: _perturbed)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="NSFW classification threshold (default: 0.5)",
    )

    # Device
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = setup_device(force_cpu=args.cpu)

    # ------------------------------------------------------------------
    # Load checkpoint and reconstruct models
    # ------------------------------------------------------------------
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    train_config = ckpt.get("config", {})
    epsilon = train_config.get("epsilon", 0.03)
    clip_model_name = train_config.get("clip_model", args.clip_model)
    image_size = train_config.get("image_size", 224)

    print(f"  epsilon    = {epsilon}")
    print(f"  clip_model = {clip_model_name}")
    print(f"  image_size = {image_size}")
    if "best_asr" in ckpt:
        print(f"  best_asr   = {ckpt['best_asr']:.1%}")
    if "epoch" in ckpt:
        print(f"  epoch      = {ckpt['epoch']}")

    # --- Generator ---
    generator = PerturbationGenerator(epsilon=epsilon, cond_dim=256).to(device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()
    print("[Model] PerturbationGenerator loaded")

    # --- Conditioner ---
    conditioner = FeatureDifferenceConditioner(embed_dim=256).to(device)
    conditioner.load_state_dict(ckpt["conditioner"])
    conditioner.eval()
    print("[Model] FeatureDifferenceConditioner loaded")

    # --- CLIP pipeline ---
    projection_path = args.projection_path
    clip_pipeline = DifferentiableCLIPPipeline(
        clip_model_name=clip_model_name,
        projection_path=projection_path,
        device=str(device),
    ).to(device)
    clip_pipeline.eval()
    print("[Model] DifferentiableCLIPPipeline loaded")

    # --- Classifier ---
    classifier = KerasClassifierBridge.from_keras(
        args.classifier_path, device=str(device)
    )
    classifier.eval()
    print("[Model] KerasClassifierBridge loaded")

    # ------------------------------------------------------------------
    # Collect input images
    # ------------------------------------------------------------------
    if args.image:
        image_paths = [Path(args.image)]
    else:
        image_dir = Path(args.image_dir)
        image_paths = sorted(
            f for f in image_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    if not image_paths:
        print("No images found.")
        return

    print(f"\nProcessing {len(image_paths)} image(s)")
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Process each image
    # ------------------------------------------------------------------
    results = []
    n_success = 0

    print(f"\n{'='*64}")
    print(f"{'Image':<30}  {'Before':>10}  {'After':>10}  {'Result':>10}")
    print(f"{'-'*64}")

    for img_path in tqdm(image_paths, desc="Applying perturbation"):
        try:
            result = apply_perturbation(
                image_path=str(img_path),
                generator=generator,
                conditioner=conditioner,
                clip_pipeline=clip_pipeline,
                classifier=classifier,
                device=device,
                image_size=image_size,
            )
        except Exception as e:
            print(f"  [ERROR] {img_path.name}: {e}")
            continue

        # Build output path preserving extension
        stem = img_path.stem
        ext = img_path.suffix
        out_path = str(Path(args.output_dir) / f"{stem}{args.suffix}{ext}")

        # Save perturbed image at original resolution
        save_perturbed_image(
            original_pil=result["original_pil"],
            perturbed_tensor_224=result["perturbed_tensor_224"],
            original_size=result["original_size"],
            output_path=out_path,
        )

        prob_b = result["prob_before"]
        prob_a = result["prob_after"]
        label_b = "NSFW" if prob_b >= args.threshold else "SFW"
        label_a = "NSFW" if prob_a >= args.threshold else "SFW"
        flipped = prob_b >= args.threshold and prob_a < args.threshold

        if flipped:
            n_success += 1
            tag = "FLIPPED"
        else:
            tag = ""

        name_display = img_path.name[:28]
        print(
            f"  {name_display:<30}"
            f"  {prob_b:>6.4f} {label_b:<4}"
            f"  {prob_a:>6.4f} {label_a:<4}"
            f"  {tag}"
        )

        results.append({
            "input": str(img_path),
            "output": out_path,
            "original_size": list(result["original_size"]),
            "prob_before": round(prob_b, 6),
            "prob_after": round(prob_a, 6),
            "label_before": label_b,
            "label_after": label_a,
            "flipped": flipped,
        })

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = len(results)
    print(f"\n{'='*64}")
    print("Summary")
    print(f"{'='*64}")
    print(f"  Total images processed : {total}")
    print(f"  NSFW -> SFW flipped    : {n_success} / {total}"
          f"  ({100*n_success/total:.1f}%)" if total > 0 else "")
    if results:
        avg_before = np.mean([r["prob_before"] for r in results])
        avg_after = np.mean([r["prob_after"] for r in results])
        print(f"  Avg NSFW prob (before)  : {avg_before:.4f}")
        print(f"  Avg NSFW prob (after)   : {avg_after:.4f}")
        print(f"  Avg prob reduction      : {avg_before - avg_after:+.4f}")
    print(f"  Output directory        : {args.output_dir}/")

    # Save JSON report
    report_path = str(Path(args.output_dir) / "results.json")
    report = {
        "checkpoint": args.checkpoint,
        "classifier": args.classifier_path,
        "threshold": args.threshold,
        "epsilon": epsilon,
        "summary": {
            "total": total,
            "flipped": n_success,
            "flip_rate": round(n_success / total, 4) if total > 0 else 0.0,
            "avg_prob_before": round(avg_before, 6) if results else 0.0,
            "avg_prob_after": round(avg_after, 6) if results else 0.0,
        },
        "images": results,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Report saved to         : {report_path}")


if __name__ == "__main__":
    main()
