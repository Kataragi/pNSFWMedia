#!/usr/bin/env python3
"""
SFM Adversarial Perturbation — Gradio GUI

Provides a web-based interface for applying adversarial perturbations
to images using trained SFM models.

Usage
-----
    python src/adversarial/gui.py

    # Custom port
    python src/adversarial/gui.py --port 7861

    # Share publicly (creates a temporary public URL)
    python src/adversarial/gui.py --share
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adversarial.models import (
    DifferentiableCLIPPipeline,
    FeatureDifferenceConditioner,
    KerasClassifierBridge,
    PerturbationGenerator,
)

# ---------------------------------------------------------------------------
# Global model cache (keeps models in VRAM)
# ---------------------------------------------------------------------------

class ModelCache:
    """
    Singleton cache for loaded models to avoid reloading on each inference.
    Models stay in VRAM until the application exits.
    """

    def __init__(self):
        self.device: Optional[torch.device] = None
        self.generators: dict = {}  # checkpoint_name -> PerturbationGenerator
        self.conditioners: dict = {}  # checkpoint_name -> FeatureDifferenceConditioner
        self.clip_pipeline: Optional[DifferentiableCLIPPipeline] = None
        self.classifier: Optional[KerasClassifierBridge] = None
        self.configs: dict = {}  # checkpoint_name -> config dict

    def setup_device(self, force_cpu: bool = False) -> torch.device:
        if self.device is not None:
            return self.device
        if force_cpu or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")
        return self.device

    def load_shared_models(
        self,
        classifier_path: str,
        projection_path: str,
        clip_model_name: str = "ViT-B/32",
    ):
        """Load classifier and CLIP pipeline (shared across all checkpoints)."""
        if self.classifier is None:
            self.classifier = KerasClassifierBridge.from_keras(
                classifier_path, device=str(self.device)
            )
            self.classifier.eval()

        if self.clip_pipeline is None:
            self.clip_pipeline = DifferentiableCLIPPipeline(
                clip_model_name=clip_model_name,
                projection_path=projection_path,
                device=str(self.device),
            ).to(self.device)
            self.clip_pipeline.eval()

    def load_checkpoint(self, checkpoint_path: str, name: str):
        """Load a specific checkpoint's generator and conditioner."""
        if name in self.generators:
            return  # Already loaded

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = ckpt.get("config", {})
        epsilon = config.get("epsilon", 0.03)

        generator = PerturbationGenerator(epsilon=epsilon, cond_dim=256).to(self.device)
        generator.load_state_dict(ckpt["generator"])
        generator.eval()

        conditioner = FeatureDifferenceConditioner(embed_dim=256).to(self.device)
        conditioner.load_state_dict(ckpt["conditioner"])
        conditioner.eval()

        self.generators[name] = generator
        self.conditioners[name] = conditioner
        self.configs[name] = config

    def get_models(self, name: str):
        """Get generator and conditioner for a specific checkpoint."""
        return self.generators.get(name), self.conditioners.get(name)


# Global cache instance
_cache = ModelCache()


# ---------------------------------------------------------------------------
# Model paths configuration
# ---------------------------------------------------------------------------

MODEL_OPTIONS = {
    "high_noise": "models/adversarial/high_noise.pt",
    "medium_noise": "models/adversarial/medium_noise.pt",
    "low_noise": "models/adversarial/low_noise.pt",
    "very_lownoise": "models/adversarial/very_lownoise.pt",
}

CLASSIFIER_PATH = "models/pnsfwmedia_classifier.keras"
PROJECTION_PATH = "models/clip_projection.pt"


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def load_image_tensor(pil_image: Image.Image, size: int = 224) -> tuple:
    """Convert PIL image to tensor."""
    original_size = pil_image.size  # (W, H)
    resized = pil_image.resize((size, size), Image.BICUBIC)
    arr = np.array(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return tensor, original_size


def apply_perturbation_to_original(
    original_pil: Image.Image,
    perturbed_tensor_224: torch.Tensor,
    original_size: tuple,
) -> Image.Image:
    """Apply 224x224 perturbation to original resolution image."""
    W, H = original_size

    # Convert original to tensor at 224x224
    orig_resized = original_pil.resize((224, 224), Image.BICUBIC)
    orig_arr = np.array(orig_resized, dtype=np.float32) / 255.0
    orig_tensor_224 = torch.from_numpy(orig_arr).permute(2, 0, 1).unsqueeze(0)

    # Compute delta at 224x224
    delta_224 = perturbed_tensor_224.cpu() - orig_tensor_224

    # Upscale delta to original resolution
    delta_full = F.interpolate(
        delta_224,
        size=(H, W),
        mode="bicubic",
        align_corners=False,
    )

    # Apply delta to original
    orig_full_arr = np.array(original_pil, dtype=np.float32) / 255.0
    orig_full_tensor = torch.from_numpy(orig_full_arr).permute(2, 0, 1).unsqueeze(0)
    perturbed_full = torch.clamp(orig_full_tensor + delta_full, 0.0, 1.0)

    # Convert back to PIL
    result_arr = (perturbed_full.squeeze(0).permute(1, 2, 0).numpy() * 255.0)
    result_arr = result_arr.clip(0, 255).astype(np.uint8)
    return Image.fromarray(result_arr)


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def process_single_model(
    image: Image.Image,
    model_name: str,
) -> tuple[Image.Image, str]:
    """
    Process image with a single model.

    Returns:
        (perturbed_image, result_text)
    """
    if image is None:
        return None, "[ERROR] No image provided"

    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Setup device and load models
    device = _cache.setup_device()

    # Load shared models
    try:
        _cache.load_shared_models(CLASSIFIER_PATH, PROJECTION_PATH)
    except Exception as e:
        return None, f"[ERROR] Failed to load shared models: {e}"

    # Load checkpoint
    checkpoint_path = MODEL_OPTIONS.get(model_name)
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None, f"[ERROR] Model not found: {checkpoint_path}"

    try:
        _cache.load_checkpoint(checkpoint_path, model_name)
    except Exception as e:
        return None, f"[ERROR] Failed to load {model_name}: {e}"

    generator, conditioner = _cache.get_models(model_name)

    # Process image
    tensor_224, original_size = load_image_tensor(image)
    tensor_224 = tensor_224.to(device)

    # Get embeddings and predictions
    orig_embed = _cache.clip_pipeline(tensor_224)
    orig_pred = _cache.classifier(orig_embed)

    # Generate perturbation
    cond = conditioner(orig_embed)
    perturbed_224 = generator(tensor_224, cond)

    # Classify perturbed
    pert_embed = _cache.clip_pipeline(perturbed_224)
    pert_pred = _cache.classifier(pert_embed)

    prob_before = orig_pred.item()
    prob_after = pert_pred.item()
    delta = prob_after - prob_before

    # Apply perturbation at original resolution
    result_image = apply_perturbation_to_original(
        image, perturbed_224, original_size
    )

    result_text = f"[{model_name}] {prob_before:.4f} -> {prob_after:.4f} ({delta:+.4f})"

    return result_image, result_text


@torch.no_grad()
def process_all_models(image: Image.Image) -> tuple:
    """
    Process image with all 4 models.

    Returns:
        (high_noise_img, medium_noise_img, low_noise_img, very_lownoise_img, result_text)
    """
    if image is None:
        return None, None, None, None, "[ERROR] No image provided"

    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Setup device and load models
    device = _cache.setup_device()

    # Load shared models
    try:
        _cache.load_shared_models(CLASSIFIER_PATH, PROJECTION_PATH)
    except Exception as e:
        return None, None, None, None, f"[ERROR] Failed to load shared models: {e}"

    # Prepare image tensor once
    tensor_224, original_size = load_image_tensor(image)
    tensor_224 = tensor_224.to(device)

    # Get original embedding and prediction once
    orig_embed = _cache.clip_pipeline(tensor_224)
    orig_pred = _cache.classifier(orig_embed)
    prob_before = orig_pred.item()

    results = []
    result_images = []

    for model_name in ["high_noise", "medium_noise", "low_noise", "very_lownoise"]:
        checkpoint_path = MODEL_OPTIONS.get(model_name)
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            result_images.append(None)
            results.append(f"[{model_name}] Model not found")
            continue

        try:
            _cache.load_checkpoint(checkpoint_path, model_name)
        except Exception as e:
            result_images.append(None)
            results.append(f"[{model_name}] Load failed: {e}")
            continue

        generator, conditioner = _cache.get_models(model_name)

        # Generate perturbation
        cond = conditioner(orig_embed)
        perturbed_224 = generator(tensor_224, cond)

        # Classify perturbed
        pert_embed = _cache.clip_pipeline(perturbed_224)
        pert_pred = _cache.classifier(pert_embed)
        prob_after = pert_pred.item()
        delta = prob_after - prob_before

        # Apply perturbation at original resolution
        result_image = apply_perturbation_to_original(
            image, perturbed_224, original_size
        )

        result_images.append(result_image)
        results.append(f"[{model_name}] {prob_before:.4f} -> {prob_after:.4f} ({delta:+.4f})")

    result_text = "\n".join(results)

    return result_images[0], result_images[1], result_images[2], result_images[3], result_text


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def create_ui():
    """Create and return the Gradio interface."""

    with gr.Blocks(title="pNSFWMedia Adversarial Perturbation") as demo:
        gr.Markdown("# pNSFWMedia Adversarial Perturbation")
        gr.Markdown("NSFW画像に知覚困難なノイズを加えてSFWに誤認させます。")

        with gr.Tabs():
            # Tab 1: Single model mode
            with gr.TabItem("Single Model"):
                with gr.Row():
                    with gr.Column():
                        input_image_single = gr.Image(
                            label="Input Image",
                            type="pil",
                        )
                        model_selector = gr.Dropdown(
                            choices=list(MODEL_OPTIONS.keys()),
                            value="high_noise",
                            label="Model",
                        )
                        run_single_btn = gr.Button("Run", variant="primary")

                    with gr.Column():
                        output_image_single = gr.Image(
                            label="Output Image",
                            type="pil",
                        )
                        result_text_single = gr.Textbox(
                            label="Result",
                            lines=2,
                        )

                run_single_btn.click(
                    fn=process_single_model,
                    inputs=[input_image_single, model_selector],
                    outputs=[output_image_single, result_text_single],
                )

            # Tab 2: All models mode
            with gr.TabItem("All Models"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image_all = gr.Image(
                            label="Input Image",
                            type="pil",
                        )
                        run_all_btn = gr.Button("Run All Models", variant="primary")
                        result_text_all = gr.Textbox(
                            label="Results",
                            lines=5,
                        )

                    with gr.Column(scale=2):
                        with gr.Row():
                            output_high = gr.Image(label="high_noise", type="pil")
                            output_medium = gr.Image(label="medium_noise", type="pil")
                        with gr.Row():
                            output_low = gr.Image(label="low_noise", type="pil")
                            output_very_low = gr.Image(label="very_lownoise", type="pil")

                run_all_btn.click(
                    fn=process_all_models,
                    inputs=[input_image_all],
                    outputs=[output_high, output_medium, output_low, output_very_low, result_text_all],
                )

        gr.Markdown("""
        ---
        ### Notes
        - モデルは初回ロード後VRAMに保持されます（連続実行時に再ロードしません）
        - 出力画像は入力画像と同じ解像度で保存されます
        - 右クリックで出力画像をダウンロードできます
        """)

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gradio GUI for SFM Adversarial Perturbation"
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port to run the server on (default: 7860)",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public shareable link",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU mode",
    )

    args = parser.parse_args()

    # Pre-setup device
    _cache.setup_device(force_cpu=args.cpu)

    # Create and launch UI
    demo = create_ui()
    demo.launch(
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
