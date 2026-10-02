"""InsightFace-backed face detection."""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis


BoundingBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class DetectedFace:
    """A detected face and its confidence score."""

    bounding_box: BoundingBox
    confidence: float


class FaceDetector:
    """Detect faces using InsightFace's SCRFD detector."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: Tuple[int, int] = (640, 640),
        ctx_id: int = -1,
    ) -> None:
        self._analysis = FaceAnalysis(name=model_name)
        # ctx_id=-1 uses CPU. The model files are downloaded by InsightFace
        # on first use and cached in ~/.insightface.
        self._analysis.prepare(ctx_id=ctx_id, det_size=det_size)

    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """Return all faces detected in a BGR OpenCV image."""
        if image is None or image.size == 0:
            raise ValueError("The supplied image is empty")

        faces = self._analysis.get(image)
        detected = []
        height, width = image.shape[:2]
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            box = (
                max(0, min(x1, width - 1)),
                max(0, min(y1, height - 1)),
                max(0, min(x2, width - 1)),
                max(0, min(y2, height - 1)),
            )
            detected.append(DetectedFace(box, float(face.det_score)))
        return detected

    @staticmethod
    def draw_bounding_boxes(
        image: np.ndarray,
        faces: List[DetectedFace],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw each detected face and confidence on a copy of the image."""
        output = image.copy()
        for index, face in enumerate(faces, start=1):
            x1, y1, x2, y2 = face.bounding_box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            label = f"Face {index}: {face.confidence:.2f}"
            text_y = max(y1 - 8, 18)
            cv2.putText(
                output,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        return output

