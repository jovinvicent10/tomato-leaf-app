"""Inference matching Tomato_Leaf_Disease_Final_Colab_OOD.ipynb."""
import io
import json
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = 20_000_000
CLASSES = ['Early_Blight', 'Healthy', 'Late_Blight']


def prepare(data, resample):
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > Image.MAX_IMAGE_PIXELS:
                raise ValueError('Please choose an image smaller than 20 megapixels.')
            image = ImageOps.exif_transpose(image).convert('RGB')
            return np.asarray(image.resize((224, 224), resample), dtype=np.float32)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError('This file could not be read as an image. Try a JPG or PNG.') from error


def normalize(features):
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if not np.isfinite(features).all() or np.any(norms < 1e-12):
        raise ValueError('The image produced invalid features. Please try another image.')
    return features / norms


class TomatoSystem:
    def __init__(self, folder):
        import tensorflow as tf
        folder = Path(folder)
        self.meta = json.loads((folder / 'metadata.json').read_text())
        if self.meta['schema_version'] != 1 or self.meta['classes'] != CLASSES:
            raise ValueError('Unsupported model bundle or class order.')
        if self.meta['image_size'] != [224, 224]:
            raise ValueError('Unsupported image dimensions.')
        with np.load(folder / 'ood_detector.npz', allow_pickle=False) as saved:
            self.center = saved['center'].copy()
            self.threshold = float(saved['threshold'])
        self.extractor = tf.keras.models.load_model(
            folder / 'ood_feature_extractor.keras', compile=False)
        self.disease = tf.keras.models.load_model(
            folder / 'best_tomato_mobilenetv2.keras', compile=False)
        if self.center.shape != (self.extractor.output_shape[-1],):
            raise ValueError('Detector and feature extractor do not match.')
        if not np.isfinite(self.center).all() or not np.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError('Invalid saved OOD detector.')
        if self.disease.output_shape[-1] != len(CLASSES):
            raise ValueError('Disease model does not match saved classes.')
        self.lock = threading.Lock()

    def predict(self, data):
        pixels = prepare(data, Image.Resampling.BILINEAR)
        with self.lock:
            features = normalize(self.extractor(pixels[None], training=False).numpy())
            distance = float(np.linalg.norm(features - self.center, axis=1)[0])
            result = dict(accepted=distance <= self.threshold,
                          distance=distance, threshold=self.threshold,
                          disease=None, confidence=None, probabilities=[])
            if not result['accepted']:
                return result
            disease_pixels = prepare(data, Image.Resampling.NEAREST)
            probs = self.disease(disease_pixels[None], training=False).numpy()[0]
        if not np.isfinite(probs).all():
            raise ValueError('Invalid disease scores. Please try another image.')
        index = int(np.argmax(probs))
        result.update(disease=CLASSES[index].replace('_', ' '), confidence=float(probs[index]),
                      probabilities=[dict(label=c.replace('_', ' '), score=float(p))
                                     for c, p in zip(CLASSES, probs)])
        return result
