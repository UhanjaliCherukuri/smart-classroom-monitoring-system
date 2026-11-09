"""
webrtc_processor.py
--------------------
WebRTC video processor for the Smart Classroom Monitoring System.

This module is ADDITIVE — it does not modify any existing file.

Design rules
------------
* No Streamlit imports.             Streamlit's thread model is incompatible
                                    with the WebRTC callback thread.
* No st.session_state access.       Session state can only be safely read/written
                                    from Streamlit's main script thread.
* No CSV writes.                    CSV logging stays entirely in app.py.
* No YOLO model loading.            The detector is created once in app.py and
                                    passed in via the constructor — avoids
                                    loading a second copy of the weights.
* Thread-safe snapshot.             `recv()` runs in a background thread managed
                                    by aiortc/streamlit-webrtc.  All shared state
                                    is protected by a threading.Lock so app.py
                                    can safely call get_snapshot() from the main
                                    Streamlit thread at any time.

Compatible with
---------------
  streamlit-webrtc == 0.79.1
  av               == 18.1.0
"""

import threading
from typing import Optional

import av
import numpy as np
from streamlit_webrtc import VideoProcessorBase

# Import the project's detector — no YOLO weights loaded here.
from detector import PersonDetector


class ClassroomVideoProcessor(VideoProcessorBase):
    """
    WebRTC video processor that runs YOLO person detection on every
    incoming browser camera frame.

    Usage in app.py
    ---------------
        processor_factory = lambda: ClassroomVideoProcessor(detector)
        ctx = webrtc_streamer(
            key="webrtc",
            video_processor_factory=processor_factory,
            ...
        )
        # Read results safely from the main thread:
        if ctx.video_processor:
            count, frames, err = ctx.video_processor.get_snapshot()
    """

    def __init__(self, detector: PersonDetector) -> None:
        """
        Parameters
        ----------
        detector : PersonDetector
            An already-initialised PersonDetector instance created in app.py.
            The same YOLO model weights are reused — no second copy is loaded.
        """
        self._detector = detector

        # --- Shared state protected by a lock ---
        # recv() runs in a background thread; get_snapshot() is called from the
        # Streamlit main thread.  The lock ensures consistent reads/writes.
        self._lock = threading.Lock()

        self._people_count:  int            = 0    # latest person count
        self._frame_count:   int            = 0    # total processed frames
        self._last_error:    Optional[str]  = None # last exception message, if any

    # ------------------------------------------------------------------
    # VideoProcessorBase interface
    # ------------------------------------------------------------------

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Called by streamlit-webrtc for every incoming video frame.

        Parameters
        ----------
        frame : av.VideoFrame
            Raw frame delivered by the browser camera via WebRTC.

        Returns
        -------
        av.VideoFrame
            Annotated frame (green bounding boxes + count overlay) sent
            back to the browser for display.

        Thread safety
        -------------
        This method runs in a background aiortc thread.  Shared attributes
        (_people_count, _frame_count, _last_error) are only updated inside
        the lock block after all CPU-bound work is done, so the lock is held
        for the minimum possible time.
        """
        # Step 1 — Convert WebRTC frame → BGR NumPy array
        # av.VideoFrame.to_ndarray() with format='bgr24' gives the same
        # layout that OpenCV and detector.detect() expect.
        bgr_frame: np.ndarray = frame.to_ndarray(format="bgr24")

        # Step 2 — Run YOLO detection (CPU-bound; outside the lock)
        try:
            annotated_bgr, count, _ = self._detector.detect(bgr_frame)
            error: Optional[str] = None
        except Exception as exc:
            # On any unexpected error keep the original frame so the
            # video stream does not crash; record the error message.
            annotated_bgr = bgr_frame
            count         = 0
            error         = f"{type(exc).__name__}: {exc}"

        # Step 3 — Update shared state under the lock (fast — no I/O here)
        with self._lock:
            self._people_count = count
            self._frame_count += 1
            self._last_error   = error

        # Step 4 — Convert annotated BGR array back to av.VideoFrame
        # Must use the same format string ('bgr24') so the pixel layout
        # matches what the browser WebRTC pipeline expects.
        out_frame = av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")

        # Carry the original presentation timestamp and time-base forward
        # so the browser decoder sees a continuous, correctly-timed stream.
        out_frame.pts      = frame.pts
        out_frame.time_base = frame.time_base

        return out_frame

    # ------------------------------------------------------------------
    # Thread-safe result accessor
    # ------------------------------------------------------------------

    def get_snapshot(self) -> tuple[int, int, Optional[str]]:
        """
        Return the latest detection results.

        Safe to call from the Streamlit main thread at any time — the
        internal lock guarantees a consistent view of the shared state.

        Returns
        -------
        people_count : int
            Number of people detected in the most recently processed frame.
        frame_count  : int
            Total number of frames processed since the processor was created.
        last_error   : str | None
            The exception message from the most recent failed frame, or None
            if the last frame processed successfully.

        Example
        -------
            count, frames, err = processor.get_snapshot()
            if err:
                st.warning(f"Detection error: {err}")
            else:
                st.write(f"{count} people detected across {frames} frames")
        """
        with self._lock:
            return self._people_count, self._frame_count, self._last_error

