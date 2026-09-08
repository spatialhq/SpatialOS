"""2D semantic segmentation of RGB keyframes, as the AI input to 3D refinement.

Runs on individual keyframes (not the point cloud directly) because mature,
easily-obtainable pretrained semantic segmentation models are 2D image
models; the labels are then back-projected into 3D using the depth+pose we
already have (see `backproject.py`). This avoids depending on specialized 3D
deep-learning stacks (Mask3D, PTv3, ...) which are heavier to install and
require task-specific pretrained checkpoints.

Requires the optional `ai` extra: `pip install "spatial-os[ai]"`.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Segmenter(Protocol):
    """Maps an RGB image to a per-pixel class-id map + an id->name lookup."""

    def segment(self, rgb: np.ndarray) -> tuple[np.ndarray, dict[int, str]]: ...


class SegformerADE20KSegmenter:
    """Pretrained Segformer (ADE20K, 150 indoor/outdoor classes) via Hugging Face.

    Lazy-imports torch/transformers so the rest of spatial-os works without
    the `ai` extra installed. CPU inference is supported (slow but workable
    for a handful of keyframes); GPU is used automatically if available.
    """

    MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"

    def __init__(self) -> None:
        try:
            import torch
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "Semantic segmentation requires the 'ai' extra: pip install \"spatial-os[ai]\""
            ) from exc

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = SegformerImageProcessor.from_pretrained(self.MODEL_ID)
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.MODEL_ID).to(self.device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def segment(self, rgb: np.ndarray) -> tuple[np.ndarray, dict[int, str]]:
        torch = self._torch
        rgb_pil_input = np.ascontiguousarray(rgb[:, :, ::-1]) if rgb.shape[-1] == 3 else rgb  # BGR (cv2) -> RGB
        inputs = self.processor(images=rgb_pil_input, return_tensors="pt").to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits  # (1, num_classes, h', w')

        upsampled = torch.nn.functional.interpolate(
            logits, size=rgb.shape[:2], mode="bilinear", align_corners=False
        )
        class_map = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.int32)
        return class_map, self.id2label


def structural_label_ids(id2label: dict[int, str]) -> dict[str, int | None]:
    """Best-effort lookup of wall/floor/ceiling class ids by name, since exact
    indices vary by checkpoint/label-set version -- match by substring
    against the model's own id2label rather than hardcoding ADE20K indices.
    """

    def find(*keywords: str) -> int | None:
        for idx, name in id2label.items():
            lname = name.lower()
            if any(kw in lname for kw in keywords):
                return idx
        return None

    return {
        "wall": find("wall"),
        "floor": find("floor"),
        "ceiling": find("ceiling"),
    }
