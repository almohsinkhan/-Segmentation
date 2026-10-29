"""
Reusable Egg Segmentation Pipeline
Grounding DINO Tiny detects eggs; SAM ViT-B generates instance masks.

Run from the project root:
    python egg_pipeline.py --input-dir data --output-dir outputs/egg_segmentation --device cpu

Optional quick test:
    python egg_pipeline.py --input-dir data/test/egg --output-dir outputs/test --max-images 3

CPU is the default because this project has a 4 GB GPU. Use --device cuda only if memory permits.
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from segment_anything import SamPredictor, sam_model_registry
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Grounding DINO + SAM egg segmentation.")
    parser.add_argument("--input-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/egg_segmentation"))
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--dino-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam-checkpoint", type=Path, default=Path("models/sam_vit_b_01ec64.pth"))
    parser.add_argument("--sam-type", choices=("vit_b", "vit_l", "vit_h"), default="vit_b")
    parser.add_argument("--prompt", default="egg.")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--max-images", type=int, default=0, help="0 means process all images.")
    parser.add_argument("--save-overlay", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def choose_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
    return requested


def discover_images(root: Path):
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    images = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No images found under {root}")
    return images


def output_stem(path: Path, input_root: Path):
    # Include parent folder names to avoid filename collisions across train/valid/test.
    return "__".join(path.relative_to(input_root).with_suffix("").parts).replace(" ", "_")


def load_models(args, device):
    print(f"Loading Grounding DINO on {device}...")
    processor = AutoProcessor.from_pretrained(args.dino_model)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(args.dino_model).to(device)
    dino.eval()

    if not args.sam_checkpoint.is_file():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {args.sam_checkpoint}\n"
            "Expected models/sam_vit_b_01ec64.pth, or pass --sam-checkpoint PATH."
        )
    print(f"Loading SAM {args.sam_type} on {device}...")
    sam = sam_model_registry[args.sam_type](checkpoint=str(args.sam_checkpoint))
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    print("Models loaded.")
    return processor, dino, predictor


def detect_eggs(image_rgb, processor, dino, device, prompt, box_threshold, text_threshold):
    inputs = processor(images=image_rgb, text=prompt, return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

    start = time.perf_counter()
    with torch.inference_mode():
        outputs = dino(**inputs)
    dino_seconds = time.perf_counter() - start

    detections = processor.post_process_grounded_object_detection(
        outputs,
        inputs.get("input_ids"),
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image_rgb.shape[:2]],
    )[0]

    boxes = detections["boxes"].detach().cpu().numpy().astype(np.float32)
    scores = detections["scores"].detach().cpu().numpy().astype(float)
    return boxes, scores, dino_seconds


def segment_eggs(image_rgb, boxes, predictor):
    predictor.set_image(image_rgb)
    masks, sam_scores = [], []
    start = time.perf_counter()

    for box in boxes:
        predicted_masks, predicted_scores, _ = predictor.predict(
            box=box.astype(np.float32),
            multimask_output=False,
        )
        masks.append(predicted_masks[0].astype(bool))
        sam_scores.append(float(predicted_scores[0]) if len(predicted_scores) else float("nan"))

    return masks, sam_scores, time.perf_counter() - start


def save_outputs(image_rgb, boxes, dino_scores, masks, sam_scores, stem, output_dir, save_overlay):
    instance_dir = output_dir / "masks"
    combined_dir = output_dir / "combined"
    overlay_dir = output_dir / "overlays"
    instance_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)

    # One binary PNG per detected egg: egg pixels=255, background=0.
    for index, mask in enumerate(masks, start=1):
        cv2.imwrite(str(instance_dir / f"{stem}_egg_{index:02d}.png"), mask.astype(np.uint8) * 255)

    # Combined binary mask: all detected eggs are white.
    combined = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    for mask in masks:
        combined[mask] = 255
    cv2.imwrite(str(combined_dir / f"{stem}_combined.png"), combined)

    if save_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image_rgb)
        for index, (box, score, mask) in enumerate(zip(boxes, dino_scores, masks), start=1):
            ax.contour(mask.astype(np.uint8), levels=[0.5], linewidths=1.5)
            x1, y1, x2, y2 = box
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=1))
            ax.text(x1, max(0, y1 - 3), f"Egg {index}: {score:.2f}",
                    fontsize=8, bbox={"alpha": 0.65})
        ax.set_title(f"Grounding DINO + SAM — {len(masks)} detections")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(overlay_dir / f"{stem}_overlay.jpg", dpi=140, bbox_inches="tight")
        plt.close(fig)


def main():
    args = parse_args()
    device = choose_device(args.device)
    images = discover_images(args.input_dir)
    if args.max_images > 0:
        images = images[:args.max_images]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor, dino, predictor = load_models(args, device)

    csv_path = args.output_dir / "results.csv"
    columns = [
        "image", "status", "detections", "dino_seconds", "sam_seconds",
        "total_seconds", "dino_confidences", "sam_predicted_iou", "error"
    ]

    overall_start = time.perf_counter()
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()

        for index, path in enumerate(images, start=1):
            print(f"\n[{index}/{len(images)}] {path}")
            row = {
                "image": str(path), "status": "failed", "detections": 0,
                "dino_seconds": "", "sam_seconds": "", "total_seconds": "",
                "dino_confidences": "", "sam_predicted_iou": "", "error": ""
            }
            image_start = time.perf_counter()

            try:
                image_bgr = cv2.imread(str(path))
                if image_bgr is None:
                    raise ValueError("OpenCV could not read the image.")
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                stem = output_stem(path, args.input_dir)

                boxes, dino_scores, dino_seconds = detect_eggs(
                    image_rgb, processor, dino, device, args.prompt,
                    args.box_threshold, args.text_threshold
                )
                masks, sam_scores, sam_seconds = segment_eggs(image_rgb, boxes, predictor)
                save_outputs(
                    image_rgb, boxes, dino_scores, masks, sam_scores,
                    stem, args.output_dir, args.save_overlay
                )

                row.update({
                    "status": "ok" if len(boxes) else "no_detections",
                    "detections": len(boxes),
                    "dino_seconds": f"{dino_seconds:.4f}",
                    "sam_seconds": f"{sam_seconds:.4f}",
                    "total_seconds": f"{time.perf_counter() - image_start:.4f}",
                    "dino_confidences": ";".join(f"{score:.4f}" for score in dino_scores),
                    "sam_predicted_iou": ";".join(f"{score:.4f}" for score in sam_scores),
                })
                print(
                    f"  detections={len(boxes)}, DINO={dino_seconds:.2f}s, "
                    f"SAM={sam_seconds:.2f}s, status={row['status']}"
                )

            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"  ERROR: {row['error']}", file=sys.stderr)

            writer.writerow(row)
            csv_file.flush()
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    print(f"\nFinished {len(images)} images in {time.perf_counter() - overall_start:.2f}s.")
    print(f"Results: {csv_path.resolve()}")
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
