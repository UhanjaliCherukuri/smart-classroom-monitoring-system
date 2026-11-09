"""
camera.py
---------
Thin wrapper around OpenCV VideoCapture.
Provides safe open / read / release with error reporting.
"""

import cv2
import numpy as np


class WebcamCapture:
    """Manages the laptop webcam via OpenCV."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self._cap: cv2.VideoCapture | None = None
        self.error: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """Open the webcam. Returns True on success."""
        try:
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                # Try without backend hint (Linux / macOS fallback)
                cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                self.error = (
                    f"Cannot open camera index {self.camera_index}. "
                    "Make sure your webcam is connected and not used by another app."
                )
                return False
            self._cap = cap
            self.error = None
            return True
        except Exception as exc:
            self.error = f"Webcam error: {exc}"
            return False

    def release(self) -> None:
        """Release the webcam resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    # Frame reading
    # ------------------------------------------------------------------

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """
        Read a single frame.

        Returns
        -------
        (success, frame)  –  frame is BGR numpy array or None on failure.
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None
        ret, frame = self._cap.read()
        if not ret:
            return False, None
        return True, frame

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.release()

