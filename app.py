#!/usr/bin/env python3
"""
pNSFWMedia — Hugging Face Spaces App

Gradio web interface for NSFW image classification and adversarial perturbation.
Models are automatically downloaded from HuggingFace Hub on first launch.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from adversarial.models import (
    DifferentiableCLIPPipeline,
    FeatureDifferenceConditioner,
    KerasClassifierBridge,
    PerturbationGenerator,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_REPO_ID = "kataragi/adversarial"

MODEL_DIR = Path("models")
ADVERSARIAL_DIR = MODEL_DIR / "adversarial"

CLASSIFIER_PATH = MODEL_DIR / "pnsfwmedia_classifier.keras"
PROJECTION_PATH = MODEL_DIR / "clip_projection.pt"

MODEL_OPTIONS = {
    "high_noise": ADVERSARIAL_DIR / "high_noise.pt",
    "medium_noise": ADVERSARIAL_DIR / "medium_noise.pt",
    "low_noise": ADVERSARIAL_DIR / "low_noise.pt",
    "very_lownoise": ADVERSARIAL_DIR / "very_lownoise.pt",
}

# Adversarial models to download from HuggingFace
# (classifier and projection are already in the Git repo)
HF_ADVERSARIAL_FILES = {
    "high_noise.pt": ADVERSARIAL_DIR / "high_noise.pt",
    "medium_noise.pt": ADVERSARIAL_DIR / "medium_noise.pt",
    "low_noise.pt": ADVERSARIAL_DIR / "low_noise.pt",
    "very_lownoise.pt": ADVERSARIAL_DIR / "very_lownoise.pt",
}


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def download_models():
    """Download adversarial models from HuggingFace Hub if not present."""
    ADVERSARIAL_DIR.mkdir(parents=True, exist_ok=True)

    for hf_filename, local_path in HF_ADVERSARIAL_FILES.items():
        if local_path.exists():
            print(f"  [OK] {local_path}")
            continue
        print(f"Downloading {hf_filename} ...")
        try:
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=hf_filename,
                local_dir=str(ADVERSARIAL_DIR),
            )
            print(f"  -> {local_path}")
        except Exception as e:
            print(f"  [WARN] Failed to download {hf_filename}: {e}")


# ---------------------------------------------------------------------------
# Global model cache
# ---------------------------------------------------------------------------

class ModelCache:
    """Singleton cache for loaded models."""

    def __init__(self):
        self.device: Optional[torch.device] = None
        self.generators: dict = {}
        self.conditioners: dict = {}
        self.clip_pipeline: Optional[DifferentiableCLIPPipeline] = None
        self.classifier: Optional[KerasClassifierBridge] = None
        self.configs: dict = {}

    def setup_device(self) -> torch.device:
        if self.device is not None:
            return self.device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        return self.device

    def load_shared_models(self):
        """Load classifier and CLIP pipeline (shared across all checkpoints)."""
        if self.classifier is None:
            self.classifier = KerasClassifierBridge.from_keras(
                str(CLASSIFIER_PATH), device=str(self.device)
            )
            self.classifier.eval()

        if self.clip_pipeline is None:
            self.clip_pipeline = DifferentiableCLIPPipeline(
                clip_model_name="ViT-B/32",
                projection_path=str(PROJECTION_PATH),
                device=str(self.device),
            ).to(self.device)
            self.clip_pipeline.eval()

    def load_checkpoint(self, model_name: str):
        """Load a specific checkpoint's generator and conditioner."""
        if model_name in self.generators:
            return

        checkpoint_path = MODEL_OPTIONS.get(model_name)
        if not checkpoint_path or not checkpoint_path.exists():
            raise FileNotFoundError(f"Model not found: {checkpoint_path}")

        ckpt = torch.load(
            str(checkpoint_path), map_location=self.device, weights_only=False
        )
        config = ckpt.get("config", {})
        epsilon = config.get("epsilon", 0.03)

        generator = PerturbationGenerator(epsilon=epsilon, cond_dim=256).to(self.device)
        generator.load_state_dict(ckpt["generator"])
        generator.eval()

        conditioner = FeatureDifferenceConditioner(embed_dim=256).to(self.device)
        conditioner.load_state_dict(ckpt["conditioner"])
        conditioner.eval()

        self.generators[model_name] = generator
        self.conditioners[model_name] = conditioner
        self.configs[model_name] = config

    def get_models(self, name: str):
        return self.generators.get(name), self.conditioners.get(name)


_cache = ModelCache()


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def load_image_tensor(pil_image: Image.Image, size: int = 224) -> tuple:
    """Convert PIL image to tensor."""
    original_size = pil_image.size  # (W, H)
    resized = pil_image.resize((size, size), Image.BICUBIC)
    arr = np.array(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor, original_size


def apply_perturbation_to_original(
    original_pil: Image.Image,
    perturbed_tensor_224: torch.Tensor,
    original_size: tuple,
) -> Image.Image:
    """Apply 224x224 perturbation to original resolution image."""
    W, H = original_size

    orig_resized = original_pil.resize((224, 224), Image.BICUBIC)
    orig_arr = np.array(orig_resized, dtype=np.float32) / 255.0
    orig_tensor_224 = torch.from_numpy(orig_arr).permute(2, 0, 1).unsqueeze(0)

    delta_224 = perturbed_tensor_224.cpu() - orig_tensor_224

    delta_full = F.interpolate(
        delta_224, size=(H, W), mode="bicubic", align_corners=False
    )

    orig_full_arr = np.array(original_pil, dtype=np.float32) / 255.0
    orig_full_tensor = torch.from_numpy(orig_full_arr).permute(2, 0, 1).unsqueeze(0)
    perturbed_full = torch.clamp(orig_full_tensor + delta_full, 0.0, 1.0)

    result_arr = (perturbed_full.squeeze(0).permute(1, 2, 0).numpy() * 255.0)
    result_arr = result_arr.clip(0, 255).astype(np.uint8)
    return Image.fromarray(result_arr)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def process_single_model(
    image: Image.Image, model_name: str
) -> tuple[Image.Image, str]:
    """Process image with a single model."""
    if image is None:
        return None, "[ERROR] No image provided"

    if image.mode != "RGB":
        image = image.convert("RGB")

    device = _cache.setup_device()

    try:
        _cache.load_shared_models()
    except Exception as e:
        return None, f"[ERROR] Failed to load shared models: {e}"

    try:
        _cache.load_checkpoint(model_name)
    except Exception as e:
        return None, f"[ERROR] Failed to load {model_name}: {e}"

    generator, conditioner = _cache.get_models(model_name)

    tensor_224, original_size = load_image_tensor(image)
    tensor_224 = tensor_224.to(device)

    orig_embed = _cache.clip_pipeline(tensor_224)
    orig_pred = _cache.classifier(orig_embed)

    cond = conditioner(orig_embed)
    perturbed_224 = generator(tensor_224, cond)

    pert_embed = _cache.clip_pipeline(perturbed_224)
    pert_pred = _cache.classifier(pert_embed)

    prob_before = orig_pred.item()
    prob_after = pert_pred.item()
    delta = prob_after - prob_before

    result_image = apply_perturbation_to_original(image, perturbed_224, original_size)
    result_text = f"[{model_name}] {prob_before:.4f} -> {prob_after:.4f} ({delta:+.4f})"

    return result_image, result_text


@torch.no_grad()
def process_all_models(image: Image.Image) -> tuple:
    """Process image with all 4 models."""
    if image is None:
        return None, None, None, None, "[ERROR] No image provided"

    if image.mode != "RGB":
        image = image.convert("RGB")

    device = _cache.setup_device()

    try:
        _cache.load_shared_models()
    except Exception as e:
        return None, None, None, None, f"[ERROR] Failed to load shared models: {e}"

    tensor_224, original_size = load_image_tensor(image)
    tensor_224 = tensor_224.to(device)

    orig_embed = _cache.clip_pipeline(tensor_224)
    orig_pred = _cache.classifier(orig_embed)
    prob_before = orig_pred.item()

    results = []
    result_images = []

    for model_name in ["high_noise", "medium_noise", "low_noise", "very_lownoise"]:
        try:
            _cache.load_checkpoint(model_name)
        except Exception as e:
            result_images.append(None)
            results.append(f"[{model_name}] Load failed: {e}")
            continue

        generator, conditioner = _cache.get_models(model_name)

        cond = conditioner(orig_embed)
        perturbed_224 = generator(tensor_224, cond)

        pert_embed = _cache.clip_pipeline(perturbed_224)
        pert_pred = _cache.classifier(pert_embed)
        prob_after = pert_pred.item()
        delta = prob_after - prob_before

        result_image = apply_perturbation_to_original(
            image, perturbed_224, original_size
        )

        result_images.append(result_image)
        results.append(
            f"[{model_name}] {prob_before:.4f} -> {prob_after:.4f} ({delta:+.4f})"
        )

    result_text = "\n".join(results)
    # Pad to 4 images
    while len(result_images) < 4:
        result_images.append(None)

    return (
        result_images[0],
        result_images[1],
        result_images[2],
        result_images[3],
        result_text,
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def create_ui():
    """Create and return the Gradio interface."""

    with gr.Blocks(title="pNSFWMedia Adversarial Perturbation") as demo:
        gr.Markdown("# pNSFWMedia Adversarial Perturbation")
        gr.Markdown(
            "NSFW画像に知覚困難なノイズを加えてSFWに誤認させます。\n\n"
            "Add imperceptible perturbations to NSFW images to misclassify them as SFW."
        )

        with gr.Tabs():
            # Tab 1: Single model
            with gr.TabItem("Single Model"):
                with gr.Row():
                    with gr.Column():
                        input_image_single = gr.Image(
                            label="Input Image", type="pil"
                        )
                        model_selector = gr.Dropdown(
                            choices=list(MODEL_OPTIONS.keys()),
                            value="high_noise",
                            label="Model",
                        )
                        run_single_btn = gr.Button("Run", variant="primary")

                    with gr.Column():
                        output_image_single = gr.Image(
                            label="Output Image", type="pil"
                        )
                        result_text_single = gr.Textbox(label="Result", lines=2)

                run_single_btn.click(
                    fn=process_single_model,
                    inputs=[input_image_single, model_selector],
                    outputs=[output_image_single, result_text_single],
                )

            # Tab 2: All models
            with gr.TabItem("All Models"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image_all = gr.Image(
                            label="Input Image", type="pil"
                        )
                        run_all_btn = gr.Button("Run All Models", variant="primary")
                        result_text_all = gr.Textbox(label="Results", lines=5)

                    with gr.Column(scale=2):
                        with gr.Row():
                            output_high = gr.Image(
                                label="high_noise", type="pil"
                            )
                            output_medium = gr.Image(
                                label="medium_noise", type="pil"
                            )
                        with gr.Row():
                            output_low = gr.Image(
                                label="low_noise", type="pil"
                            )
                            output_very_low = gr.Image(
                                label="very_lownoise", type="pil"
                            )

                run_all_btn.click(
                    fn=process_all_models,
                    inputs=[input_image_all],
                    outputs=[
                        output_high,
                        output_medium,
                        output_low,
                        output_very_low,
                        result_text_all,
                    ],
                )

        gr.Markdown(
            "---\n"
            "### Notes\n"
            "- モデルは初回ロード後メモリに保持されます\n"
            "- 出力画像は入力画像と同じ解像度です\n"
            "- 右クリックで出力画像をダウンロードできます"
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Downloading models (if needed) ...")
    download_models()

    _cache.setup_device()

    demo = create_ui()
    demo.launch()
