"""
detector.py
-----------
Handles YOLO model loading and person detection.
Only detects the 'person' class (class ID 0 in COCO dataset).
"""

import cv2
import numpy as np

# YOLO is imported lazily to give a clear error if not installed
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class PersonDetector:
    """
    Wraps a pretrained YOLO model and exposes a simple detect() method
    that returns annotated frames + person count.
    """

    # COCO class ID for 'person'
    PERSON_CLASS_ID = 0

    def __init__(self, model_path: str = "yolo11n.pt", confidence: float = 0.4):
        """
        Parameters
        ----------
        model_path  : Pretrained YOLO weights file (downloaded automatically
                      by Ultralytics on first run if not already cached).
        confidence  : Minimum confidence score to accept a detection.
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            )

        self.confidence = confidence
        self.model = None
        self.load_error: str | None = None
        self._load_model(model_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str) -> None:
        """Load the YOLO model; capture any error for the UI to display."""
        try:
            self.model = YOLO(model_path)
            # Warm-up: run a blank frame so the first real frame is fast
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False)
        except Exception as exc:
            self.load_error = str(exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.load_error is None

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, int, list[dict]]:
        """
        Run inference on a single BGR frame.

        Returns
        -------
        annotated_frame : Frame with bounding boxes drawn.
        count           : Number of people detected.
        detections      : List of dicts with keys: bbox, confidence.
        """
        if not self.is_ready:
            return frame, 0, []

        results = self.model.predict(
            frame,
            conf=self.confidence,
            classes=[self.PERSON_CLASS_ID],  # persons only
            verbose=False,
        )

        detections = []
        annotated = frame.copy()

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id != self.PERSON_CLASS_ID:
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)

                # Confidence label
                label = f"Person {conf:.0%}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                lx, ly = x1, max(y1 - 6, label_size[1] + 4)
                cv2.rectangle(
                    annotated,
                    (lx, ly - label_size[1] - 4),
                    (lx + label_size[0] + 4, ly + 2),
                    (0, 200, 0),
                    -1,
                )
                cv2.putText(
                    annotated, label, (lx + 2, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
                )

                detections.append({"bbox": (x1, y1, x2, y2), "confidence": conf})

        count = len(detections)

        # Overlay person count on the frame
        cv2.putText(
            annotated,
            f"People: {count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2, cv2.LINE_AA,
        )

        return annotated, count, detections

