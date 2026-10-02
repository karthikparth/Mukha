"""Utilities for validating, loading, and saving images."""

from pathlib import Path
from typing import Union

import cv2
import numpy as np


ImagePath = Union[str, Path]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def validate_image_path(path: ImagePath) -> Path:
    """Return a validated image path.

    The extension check is intentionally case-insensitive and happens before
    OpenCV reads the file so unsupported formats fail with a useful message.
    """
    image_path = Path(path)
    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported image format '{image_path.suffix}'. Use: {supported}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image_path


def load_image(path: ImagePath) -> np.ndarray:
    """Load an image as an OpenCV BGR array."""
    image_path = validate_image_path(path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {image_path}")
    return image


def save_image(image: np.ndarray, path: ImagePath) -> Path:
    """Save an OpenCV BGR image and return the destination path."""
    output_path = Path(path)
    if output_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported output format '{output_path.suffix}'. Use: {supported}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Could not write image: {output_path}")
    return output_path

