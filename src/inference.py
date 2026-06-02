"""
Inference pipeline for SUPARCO Lunar Soil Analysis.
Provides a unified API for composition prediction and terrain classification.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T

from src.model import (
    ELEMENTS, TERRAIN_CLASSES,
    load_composition_model, load_terrain_model,
)
from src.dataset import ELEMENT_FULL, ELEMENT_UNITS

_IMG_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _to_tensor(image):
    """Accept PIL Image, numpy array, or file path → normalised tensor (1,3,H,W)."""
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8)).convert('RGB')
    elif isinstance(image, Image.Image):
        image = image.convert('RGB')
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    return _IMG_TRANSFORM(image).unsqueeze(0)


class LunarAnalysisPipeline:
    """
    Full inference pipeline.
    Loads composition and terrain models once, then exposes predict() method.
    """

    def __init__(
        self,
        composition_model_path=None,
        terrain_model_path=None,
        device=None,
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.comp_model = None
        self.label_mean = None
        self.label_std = None
        self.terrain_model = None

        if composition_model_path and Path(composition_model_path).exists():
            self.comp_model, self.label_mean, self.label_std = load_composition_model(
                composition_model_path, self.device
            )

        if terrain_model_path and Path(terrain_model_path).exists():
            self.terrain_model = load_terrain_model(terrain_model_path, self.device)

    @property
    def composition_ready(self):
        return self.comp_model is not None

    @property
    def terrain_ready(self):
        return self.terrain_model is not None

    def predict_composition(self, image):
        """
        Predict heavy metal concentrations from a soil image.

        Returns dict with keys:
          predictions: {element: float}  — predicted mg/kg values
          element_info: {element: {full_name, unit, value, mean, std}}
        """
        if not self.composition_ready:
            raise RuntimeError("Composition model not loaded.")

        tensor = _to_tensor(image).to(self.device)
        with torch.no_grad():
            raw = self.comp_model(tensor).squeeze(0)  # (6,)

        # De-normalise
        values = (raw * self.label_std + self.label_mean).cpu().numpy()
        values = np.clip(values, 0, None)  # concentrations can't be negative

        result = {}
        for i, elem in enumerate(ELEMENTS):
            result[elem] = float(values[i])

        return {
            'predictions': result,
            'element_info': {
                e: {
                    'full_name': ELEMENT_FULL[e],
                    'unit': ELEMENT_UNITS[e],
                    'value': result[e],
                }
                for e in ELEMENTS
            },
        }

    def predict_terrain(self, image):
        """
        Classify terrain type from an image.

        Returns dict with keys:
          class_name: str
          confidence: float (0–1)
          all_probs: {class_name: float}
        """
        if not self.terrain_ready:
            raise RuntimeError("Terrain model not loaded.")

        tensor = _to_tensor(image).to(self.device)
        with torch.no_grad():
            logits = self.terrain_model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        return {
            'class_name': TERRAIN_CLASSES[pred_idx],
            'confidence': float(probs[pred_idx]),
            'all_probs': {TERRAIN_CLASSES[i]: float(probs[i]) for i in range(len(TERRAIN_CLASSES))},
        }

    def predict_full(self, image):
        """Run both composition and terrain prediction."""
        result = {}
        if self.composition_ready:
            result['composition'] = self.predict_composition(image)
        if self.terrain_ready:
            result['terrain'] = self.predict_terrain(image)
        return result
