"""
app.py
------
Streamlit dashboard for the Smart Classroom Monitoring System.

Run with:
    streamlit run app.py

Architecture note
-----------------
Streamlit reruns the entire script on every user interaction.
To serve a live video feed we use the "one frame per rerun" pattern:
  1. Read ONE frame from the webcam stored in session_state.
  2. Run YOLO detection on it.
  3. Update the UI.
  4. Call st.rerun() to trigger the next frame.
This allows Streamlit to process button clicks (Start / Stop) between frames.
"""

import csv
import os
import datetime
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Local modules
from camera import WebcamCapture
from detector import PersonDetector

# WebRTC support (streamlit-webrtc + processor defined in webrtc_processor.py)
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from webrtc_processor import ClassroomVideoProcessor

# ---------------------------------------------------------------------------
# CSV logging helpers  — UNCHANGED
# ---------------------------------------------------------------------------
CSV_PATH    = os.path.join(os.path.dirname(__file__), "occupancy_history.csv")
CSV_HEADERS = ["Timestamp", "People Count", "Capacity", "Occupancy %", "Status"]


def ensure_csv() -> None:
    """Create the CSV file with headers if it does not already exist."""
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADERS)


def log_to_csv(count: int, cap: int, pct: float, status: str) -> None:
    """Append one row to the CSV. Called once per processed frame, never on bare reruns."""
    timestamp    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plain_status = status.split(" ", 1)[-1] if status[0] in "🟢🟡🔴" else status
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([timestamp, count, cap, f"{pct:.1f}", plain_status])


def read_csv_tail(n: int = 20) -> pd.DataFrame:
    """Return the last n rows of the CSV as a DataFrame, newest first."""
    ensure_csv()
    try:
        df = pd.read_csv(CSV_PATH)
        return df.tail(n).iloc[::-1].reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=CSV_HEADERS)


def csv_bytes() -> bytes:
    """Return the full CSV file as bytes for the download button."""
    ensure_csv()
    with open(CSV_PATH, "rb") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Classroom Monitor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design system — Professional Light Blue & White Theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ═══════════════════════════════════════════════════════════════
   GLOBAL — light background, navy text
═══════════════════════════════════════════════════════════════ */
.stApp {
    background-color: #F0F4F8;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    color: #1E3A5F;
}
/* Hide default Streamlit chrome (keep header so sidebar toggle works) */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* ═══════════════════════════════════════════════════════════════
   SIDEBAR — light blue-grey
═══════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: #EBF0F8;
    border-right: 1px solid #C8D8EE;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] label {
    color: #1E3A5F !important;
}
[data-testid="stSidebar"] hr {
    border-top: 1px solid #C8D8EE !important;
}
[data-testid="stSidebar"] .stCaption p {
    color: #6B8DB5 !important;
    font-size: 0.72rem !important;
}
/* Sidebar number_input / slider labels */
[data-testid="stSidebar"] .stSlider label p,
[data-testid="stSidebar"] .stNumberInput label p {
    color: #1E3A5F !important;
    font-weight: 600;
}

/* ═══════════════════════════════════════════════════════════════
   SIDEBAR NAV BUTTONS — styled to look like nav pills
═══════════════════════════════════════════════════════════════ */
/* All sidebar buttons share base nav-item styling */
[data-testid="stSidebar"] [data-testid="baseButton-secondary"] {
    background:    transparent   !important;
    color:         #2D5A9E       !important;
    border:        none          !important;
    border-radius: 8px           !important;
    font-size:     0.88rem       !important;
    font-weight:   600           !important;
    text-align:    left          !important;
    padding:       0.55rem 0.9rem !important;
    width:         100%          !important;
    margin-bottom: 2px           !important;
    transition:    background 0.15s !important;
}
[data-testid="stSidebar"] [data-testid="baseButton-secondary"]:hover {
    background: #D6E4F7 !important;
}
/* Active nav page button — blue fill */
[data-testid="stSidebar"] [data-testid="baseButton-primary"] {
    background:    #2563EB !important;
    color:         white   !important;
    border:        none    !important;
    border-radius: 8px     !important;
    font-size:     0.88rem !important;
    font-weight:   600     !important;
    text-align:    left    !important;
    width:         100%    !important;
    margin-bottom: 2px     !important;
}
[data-testid="stSidebar"] [data-testid="baseButton-primary"]:hover {
    background: #1D4ED8 !important;
}

/* ═══════════════════════════════════════════════════════════════
   SIDEBAR MONITORING CONTROL BUTTONS
═══════════════════════════════════════════════════════════════ */
/* Start button — bright blue (uses data-key to scope) */
[data-testid="stSidebar"] button[kind="primary"],
[data-testid="stSidebar"] .sidebar-start-btn button {
    background-color: #2563EB !important;
    color:            white   !important;
    border:           none    !important;
    font-weight:      700     !important;
    border-radius:    8px     !important;
}
/* Stop button wrapper — red */
[data-testid="stSidebar"] .sidebar-stop-btn button {
    background-color: #DC2626 !important;
    color:            white   !important;
    border:           none    !important;
    font-weight:      700     !important;
    border-radius:    8px     !important;
}
[data-testid="stSidebar"] .sidebar-stop-btn button:hover {
    background-color: #B91C1C !important;
}

/* ═══════════════════════════════════════════════════════════════
   SECTION LABEL (sidebar config heading)
═══════════════════════════════════════════════════════════════ */
.nav-icon { font-size: 1rem; }

/* ═══════════════════════════════════════════════════════════════
   HEADER BANNER
═══════════════════════════════════════════════════════════════ */
.hero-bar {
    background:      linear-gradient(135deg, #1E3A8A 0%, #2563EB 60%, #3B82F6 100%);
    color:           white;
    padding:         1.4rem 2rem;
    border-radius:   14px;
    margin-bottom:   1.4rem;
    display:         flex;
    align-items:     center;
    justify-content: space-between;
    box-shadow:      0 4px 20px rgba(37, 99, 235, 0.30);
}
.hero-left     { display:flex; align-items:center; gap:1rem; }
.hero-icon     { font-size:2.4rem; line-height:1; }
.hero-title    { font-size:1.6rem; font-weight:800; margin:0; letter-spacing:-0.3px; }
.hero-sub      { font-size:0.82rem; opacity:0.80; margin:0.18rem 0 0; }
.hero-right    { display:flex; align-items:center; gap:1.2rem; }
.hero-datetime { font-size:0.80rem; opacity:0.80; text-align:right; }
.system-badge  {
    display:        flex;
    align-items:    center;
    gap:            0.4rem;
    background:     rgba(255,255,255,0.15);
    border:         1px solid rgba(255,255,255,0.30);
    border-radius:  999px;
    padding:        0.32rem 0.9rem;
    font-size:      0.78rem;
    font-weight:    700;
    letter-spacing: 0.5px;
}
.dot-green  { width:9px; height:9px; border-radius:50%; background:#22C55E;
              box-shadow:0 0 6px rgba(34,197,94,0.80); flex-shrink:0; }
.dot-grey   { width:9px; height:9px; border-radius:50%; background:#94A3B8; flex-shrink:0; }

/* ═══════════════════════════════════════════════════════════════
   TOP METRIC CARDS ROW
═══════════════════════════════════════════════════════════════ */
.top-card {
    background:    white;
    border-radius: 12px;
    padding:       1rem 1.2rem;
    border:        1px solid #D1E0F5;
    box-shadow:    0 2px 10px rgba(37,99,235,0.08);
    display:       flex;
    align-items:   center;
    gap:           0.9rem;
    margin-bottom: 1rem;
}
.top-card-icon {
    width:         48px;
    height:        48px;
    border-radius: 10px;
    background:    #EFF6FF;
    display:       flex;
    align-items:   center;
    justify-content: center;
    font-size:     1.5rem;
    flex-shrink:   0;
}
.top-card-label {
    font-size:      0.69rem;
    font-weight:    700;
    color:          #6B8DB5;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin:         0 0 0.2rem;
}
.top-card-value {
    font-size:   1.95rem;
    font-weight: 800;
    color:       #1E3A5F;
    margin:      0;
    line-height: 1;
}
.top-card-unit {
    font-size:   1rem;
    font-weight: 500;
    color:       #6B8DB5;
    margin-left: 2px;
}

/* ═══════════════════════════════════════════════════════════════
   SECTION LABELS
═══════════════════════════════════════════════════════════════ */
.section-label {
    font-size:      0.69rem;
    font-weight:    700;
    letter-spacing: 1.9px;
    text-transform: uppercase;
    color:          #2563EB;
    margin:         0 0 0.55rem;
    padding-bottom: 0.4rem;
    border-bottom:  2px solid #BFDBFE;
}

/* ═══════════════════════════════════════════════════════════════
   VIDEO PANEL — white card wrapping the feed
═══════════════════════════════════════════════════════════════ */
.video-panel {
    background:    white;
    border-radius: 14px;
    border:        1px solid #D1E0F5;
    box-shadow:    0 2px 12px rgba(37,99,235,0.08);
    overflow:      hidden;
    padding:       0.75rem;
}
.video-panel-header {
    display:         flex;
    justify-content: space-between;
    align-items:     center;
    margin-bottom:   0.6rem;
    padding:         0 0.2rem;
}
.video-panel-title {
    font-size:   0.88rem;
    font-weight: 700;
    color:       #1E3A5F;
}
.live-badge {
    background:    #DCFCE7;
    color:         #16A34A;
    border:        1px solid #86EFAC;
    border-radius: 999px;
    font-size:     0.72rem;
    font-weight:   700;
    padding:       0.2rem 0.65rem;
    letter-spacing: 0.5px;
}
.standby-badge {
    background:    #F1F5F9;
    color:         #94A3B8;
    border:        1px solid #CBD5E1;
    border-radius: 999px;
    font-size:     0.72rem;
    font-weight:   700;
    padding:       0.2rem 0.65rem;
    letter-spacing: 0.5px;
}
.people-count-bar {
    background:    #EFF6FF;
    border-radius: 8px;
    padding:       0.4rem 0.8rem;
    margin-top:    0.5rem;
    font-size:     0.83rem;
    font-weight:   700;
    color:         #2563EB;
    text-align:    center;
}

/* ═══════════════════════════════════════════════════════════════
   VIDEO IDLE PLACEHOLDER
═══════════════════════════════════════════════════════════════ */
.video-idle {
    background:    #F8FAFF;
    border:        2px dashed #BFDBFE;
    border-radius: 12px;
    padding:       4rem 2rem;
    text-align:    center;
    color:         #6B8DB5;
}
.video-idle-icon  { font-size:3.4rem; margin-bottom:0.6rem; line-height:1; }
.video-idle-title { font-size:1rem; font-weight:700; color:#1E3A5F; margin-bottom:0.35rem; }
.video-idle-sub   { font-size:0.85rem; color:#6B8DB5; }

/* ═══════════════════════════════════════════════════════════════
   RIGHT PANEL — classroom status card
═══════════════════════════════════════════════════════════════ */
.status-card {
    background:    white;
    border-radius: 14px;
    border:        1px solid #D1E0F5;
    box-shadow:    0 2px 12px rgba(37,99,235,0.08);
    padding:       1.2rem 1.3rem;
    margin-bottom: 1rem;
}
.status-card-title {
    font-size:   0.95rem;
    font-weight: 800;
    color:       #1E3A5F;
    margin:      0 0 0.75rem;
}
.metric-row {
    background:    #F8FAFF;
    border-radius: 8px;
    padding:       0.55rem 0.85rem;
    margin-bottom: 0.5rem;
    display:       flex;
    justify-content: space-between;
    align-items:   center;
}
.metric-row-label {
    font-size:  0.80rem;
    font-weight: 600;
    color:      #6B8DB5;
}
.metric-row-value {
    font-size:   0.95rem;
    font-weight: 800;
    color:       #1E3A5F;
}

/* ═══════════════════════════════════════════════════════════════
   STATUS PILL
═══════════════════════════════════════════════════════════════ */
.status-pill {
    display:        inline-block;
    padding:        0.38rem 1.1rem;
    border-radius:  999px;
    font-size:      0.85rem;
    font-weight:    700;
    margin-bottom:  0.75rem;
    letter-spacing: 0.2px;
}
.status-ok   { background:#DCFCE7; color:#16A34A; border:1.5px solid #86EFAC; }
.status-warn { background:#FEF9C3; color:#CA8A04; border:1.5px solid #FDE047; }
.status-full { background:#FEE2E2; color:#DC2626; border:1.5px solid #FCA5A5; }
.status-off  { background:#F1F5F9; color:#64748B; border:1.5px solid #CBD5E1; }

/* ═══════════════════════════════════════════════════════════════
   OCCUPANCY PROGRESS BAR
═══════════════════════════════════════════════════════════════ */
.progress-wrapper { margin: 0.5rem 0 0.8rem; }
.progress-track {
    background:    #E2EDFB;
    border-radius: 999px;
    height:        12px;
    overflow:      hidden;
}
.progress-fill {
    height:        100%;
    border-radius: 999px;
    transition:    width 0.25s ease;
}
.progress-label-row {
    display:         flex;
    justify-content: space-between;
    margin-top:      0.3rem;
}
.progress-label {
    font-size:   0.68rem;
    font-weight: 600;
    color:       #6B8DB5;
}

/* ═══════════════════════════════════════════════════════════════
   START / STOP BUTTONS (main area)
═══════════════════════════════════════════════════════════════ */
div[data-testid="stHorizontalBlock"] [data-testid="baseButton-primary"] {
    background-color: #2563EB !important;
    color:            white   !important;
    border:           none    !important;
    font-weight:      700     !important;
    border-radius:    8px     !important;
    font-size:        0.88rem !important;
}
div[data-testid="stHorizontalBlock"] [data-testid="baseButton-primary"]:hover {
    background-color: #1D4ED8 !important;
}
/* Stop button — red */
.stop-btn-wrap button {
    background-color: #DC2626 !important;
    color:            white   !important;
    border:           none    !important;
    font-weight:      700     !important;
    border-radius:    8px     !important;
    font-size:        0.88rem !important;
}
.stop-btn-wrap button:hover {
    background-color: #B91C1C !important;
}

/* ═══════════════════════════════════════════════════════════════
   BOTTOM SECTION — chart & table cards
═══════════════════════════════════════════════════════════════ */
.bottom-card {
    background:    white;
    border-radius: 14px;
    border:        1px solid #D1E0F5;
    box-shadow:    0 2px 12px rgba(37,99,235,0.08);
    padding:       1rem 1.2rem;
    margin-top:    0.2rem;
}
.bottom-card-title {
    font-size:    0.88rem;
    font-weight:  800;
    color:        #1E3A5F;
    margin-bottom: 0.6rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state defaults  — UNCHANGED
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "monitoring":                 False,
    "people_count":               0,
    "history":                    [],   # list of {"Time": str, "People": int}
    "frame_count":                0,
    "last_logged_frame":          -1,   # tracks which frame was last written to CSV
    "cam":                        None, # WebcamCapture instance (kept open between reruns)
    "detector":                   None,
    "detector_conf":              None,
    "active_page":                "Dashboard",   # tracks which nav page is selected
    "camera_source":              "Local Webcam (OpenCV)",  # "Local Webcam (OpenCV)" or "Browser Webcam (WebRTC)"
    # WebRTC-specific: last processor frame index consumed by the main thread
    "webrtc_last_consumed_frame": -1,
    # WebRTC-specific: last people count from a successful (error-free) frame
    "webrtc_last_valid_count":    0,
    # WebRTC-specific: True while waiting for browser to confirm stop on source switch
    "webrtc_stopping":            False,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

ensure_csv()

# ---------------------------------------------------------------------------
# Sidebar — logo, navigation buttons, configuration, monitoring controls
# ---------------------------------------------------------------------------
st.sidebar.markdown("""
<div style="text-align:center; padding:0.8rem 0 1rem;">
    <div style="font-size:2.8rem; line-height:1;">🎓</div>
    <div style="color:#1E3A5F; font-weight:800; font-size:1rem;
                letter-spacing:0.3px; margin-top:0.5rem;">Classroom Monitor</div>
    <div style="color:#6B8DB5; font-size:0.70rem;
                margin-top:0.2rem; letter-spacing:0.5px;">SMART OCCUPANCY SYSTEM</div>
</div>
""", unsafe_allow_html=True)

# --- Navigation — real Streamlit buttons, one per page ---
# The active page gets type="primary" (blue); others get type="secondary" (plain).
_pages = [
    ("📊", "Dashboard"),
    ("📈", "Analytics"),
    ("📋", "History"),
    ("⚙️",  "Settings"),
]
for _icon, _page in _pages:
    _is_active = st.session_state.active_page == _page
    if st.sidebar.button(
        f"{_icon}  {_page}",
        key=f"nav_{_page}",
        use_container_width=True,
        type="primary" if _is_active else "secondary",
    ):
        # Switching pages must NOT touch monitoring, cam, or detector
        st.session_state.active_page = _page
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<p style="color:#6B8DB5;font-size:0.72rem;font-weight:700;'
    'letter-spacing:1px;text-transform:uppercase;margin-bottom:0.5rem;">Configuration</p>',
    unsafe_allow_html=True,
)

# --- Camera source selector ---
# Switching source releases any open OpenCV webcam so it is not held idle.
_source_options = ["Local Webcam (OpenCV)", "Browser Webcam (WebRTC)"]
camera_source = st.sidebar.radio(
    "Camera Source",
    options=_source_options,
    index=_source_options.index(st.session_state.camera_source),
    key="camera_source_radio",
    help="Choose how to access the camera.",
)
# If the user switches sources, release any open OpenCV webcam immediately
# so it does not remain locked while WebRTC is active (and vice versa).
if camera_source != st.session_state.camera_source:
    if st.session_state.cam is not None:
        st.session_state.cam.release()
        st.session_state.cam = None
    # If switching AWAY from WebRTC, keep the component rendered for one more
    # rerun so desired_playing_state=False reaches the browser before unmounting.
    if st.session_state.camera_source == "Browser Webcam (WebRTC)":
        st.session_state.webrtc_stopping = True
    st.session_state.camera_source = camera_source
    st.session_state.monitoring    = False   # stop monitoring on source change

capacity = st.sidebar.number_input(
    "Classroom Capacity",
    min_value=1, max_value=200, value=30, step=1,
    help="Maximum number of students the classroom can hold.",
)

confidence_threshold = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.10, max_value=0.90, value=0.40, step=0.05,
    help="Minimum confidence score for a person to be counted.",
)

# Camera Index only makes sense for the local OpenCV webcam.
# Hide it when Browser WebRTC is selected.
if camera_source == "Local Webcam (OpenCV)":
    camera_index = st.sidebar.number_input(
        "Camera Index", min_value=0, max_value=5, value=0, step=1,
        help="0 = built-in webcam. Try 1 or 2 for external webcams.",
    )
else:
    camera_index = 0   # unused for WebRTC; set a safe default

# --- Monitoring controls — in sidebar, below Configuration ---
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<p style="color:#6B8DB5;font-size:0.72rem;font-weight:700;'
    'letter-spacing:1px;text-transform:uppercase;margin-bottom:0.5rem;">Controls</p>',
    unsafe_allow_html=True,
)

# Start button — blue (primary type inherits the primary CSS above)
# For WebRTC the button just sets monitoring=True; webrtc_streamer handles the stream.
start_btn = st.sidebar.button(
    "▶  Start Monitoring",
    key="sidebar_start_btn",
    use_container_width=True,
    type="primary",
)

# Stop button — red
st.sidebar.markdown('<div class="sidebar-stop-btn">', unsafe_allow_html=True)
stop_btn = st.sidebar.button(
    "⏹  Stop Monitoring",
    key="sidebar_stop_btn",
    use_container_width=True,
)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# --- Handle monitoring button clicks ---
if start_btn:
    # Guard: if a WebRTC stop transition is in progress, do not start a new
    # monitoring session yet.  The transition clears itself within one rerun
    # once the browser confirms the peer connection has stopped.
    if st.session_state.webrtc_stopping:
        st.sidebar.warning("⏳ Waiting for WebRTC to stop — please try again in a moment.")
    else:
        st.session_state.monitoring  = True
        st.session_state.history     = []
        st.session_state.frame_count = 0

        # Only open the OpenCV webcam here; WebRTC opens its own stream automatically
        if camera_source == "Local Webcam (OpenCV)":
            if st.session_state.cam is None or not st.session_state.cam.is_open:
                _cam = WebcamCapture(camera_index=int(camera_index))
                if _cam.open():
                    st.session_state.cam = _cam
                else:
                    st.session_state.monitoring = False
                    st.sidebar.error(f"❌ Webcam error: {_cam.error}")

if stop_btn:
    st.session_state.monitoring = False
    # Release OpenCV webcam if it is open (WebRTC stream stops via its own component)
    if st.session_state.cam is not None:
        st.session_state.cam.release()
        st.session_state.cam = None

st.sidebar.markdown("---")
st.sidebar.caption("🎓 Smart Classroom Monitoring System v1.0")

# ---------------------------------------------------------------------------
# Hero Header — title, subtitle, system status, date/time
# ---------------------------------------------------------------------------
_now_str    = datetime.datetime.now().strftime("%B %d, %Y  |  %I:%M %p")
_sys_online = st.session_state.monitoring
_dot_cls    = "dot-green" if _sys_online else "dot-grey"
_sys_label  = "System Online" if _sys_online else "System Standby"

st.markdown(f"""
<div class="hero-bar">
    <div class="hero-left">
        <div class="hero-icon">🎓</div>
        <div>
            <p class="hero-title">Smart Classroom Monitoring System</p>
            <p class="hero-sub">Real-Time Occupancy Analytics &nbsp;·&nbsp; Powered by YOLO11n</p>
        </div>
    </div>
    <div class="hero-right">
        <div class="hero-datetime">{_now_str}</div>
        <div class="system-badge">
            <div class="{_dot_cls}"></div>
            {_sys_label}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helper: occupancy logic  — UNCHANGED
# ---------------------------------------------------------------------------
def compute_occupancy(count: int, cap: int) -> tuple[float, str, str]:
    """Return (pct, status_label, css_class)."""
    pct = min(count / max(cap, 1) * 100, 100.0)
    if count == 0:
        return pct, "🟢 Empty", "status-ok"
    if count >= cap:
        return pct, "🔴 Capacity Reached", "status-full"
    if pct >= 80:
        return pct, "🟡 Almost Full", "status-warn"
    return pct, "🟢 Occupied", "status-ok"


# ---------------------------------------------------------------------------
# UI HTML generators — updated for light theme
# ---------------------------------------------------------------------------
def _top_card(icon: str, label: str, value, unit: str = "") -> str:
    """Horizontal metric card for the top row."""
    return (
        f'<div class="top-card">'
        f'  <div class="top-card-icon">{icon}</div>'
        f'  <div>'
        f'    <p class="top-card-label">{label}</p>'
        f'    <p class="top-card-value">{value}'
        f'      <span class="top-card-unit">{unit}</span>'
        f'    </p>'
        f'  </div>'
        f'</div>'
    )


def _status_pill(status: str, css_class: str) -> str:
    """Return HTML for a rounded status pill/badge."""
    return f'<div class="status-pill {css_class}">{status}</div>'


def _progress_bar(pct: float) -> str:
    """Return HTML for a styled occupancy progress bar."""
    pct_clamped = min(pct, 100.0)
    if pct_clamped >= 100:
        fill = "#EF4444"   # red
    elif pct_clamped >= 80:
        fill = "#F59E0B"   # amber
    else:
        fill = "#2563EB"   # blue (normal)
    return (
        f'<div class="progress-wrapper">'
        f'  <div class="progress-track">'
        f'    <div class="progress-fill" style="width:{pct_clamped:.1f}%;background:{fill};"></div>'
        f'  </div>'
        f'  <div class="progress-label-row">'
        f'    <span class="progress-label">0%</span>'
        f'    <span class="progress-label">{pct_clamped:.1f}% capacity used</span>'
        f'    <span class="progress-label">100%</span>'
        f'  </div>'
        f'</div>'
    )


def _metric_row(label: str, value) -> str:
    """A compact label-value row inside the status card."""
    return (
        f'<div class="metric-row">'
        f'  <span class="metric-row-label">{label}</span>'
        f'  <span class="metric-row-value">{value}</span>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# TOP METRIC CARDS — 4 columns across full width (shown on all pages)
# ---------------------------------------------------------------------------
_count_now = st.session_state.people_count
_pct_idle, _, _ = compute_occupancy(_count_now, int(capacity))
_avail_now = max(int(capacity) - _count_now, 0)

tc1, tc2, tc3, tc4 = st.columns(4, gap="medium")
with tc1:
    st.markdown(_top_card("👥", "People Detected",     _count_now),          unsafe_allow_html=True)
with tc2:
    st.markdown(_top_card("🏫", "Classroom Capacity",  int(capacity)),        unsafe_allow_html=True)
with tc3:
    st.markdown(_top_card("📊", "Occupancy",           f"{_pct_idle:.1f}", "%"), unsafe_allow_html=True)
with tc4:
    st.markdown(_top_card("🪑", "Available Seats",     _avail_now),           unsafe_allow_html=True)

# ===========================================================================
# PAGE ROUTER — render the correct page based on session_state.active_page
# ===========================================================================
_page = st.session_state.active_page

# ---------------------------------------------------------------------------
# PAGE: DASHBOARD  (default — live feed + status card + charts + log)
# ---------------------------------------------------------------------------
if _page == "Dashboard":

    # --- Main two-column layout: video feed (left) | status (right) ---
    col_video, col_status = st.columns([3, 2], gap="large")

    with col_video:
        st.markdown('<p class="section-label">📹&nbsp; Live Feed</p>', unsafe_allow_html=True)
        _feed_badge = (
            '<span class="live-badge">● LIVE</span>'
            if st.session_state.monitoring
            else '<span class="standby-badge">○ STANDBY</span>'
        )
        st.markdown(
            f'<div class="video-panel-header">'
            f'  <span class="video-panel-title">📷 Webcam Feed</span>'
            f'  {_feed_badge}'
            f'</div>',
            unsafe_allow_html=True,
        )
        video_placeholder     = st.empty()
        count_bar_placeholder = st.empty()

    with col_status:
        st.markdown('<p class="section-label">📊&nbsp; Classroom Status</p>', unsafe_allow_html=True)
        st.markdown('<div class="status-card"><p class="status-card-title">Occupancy Overview</p>',
                    unsafe_allow_html=True)
        metric_status   = st.empty()
        metric_rows     = st.empty()
        progress_bar    = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

        # Hidden placeholders (required by render_metrics signature)
        metric_people   = st.empty()
        metric_capacity = st.empty()
        metric_avail    = st.empty()
        metric_pct      = st.empty()

        st.markdown(
            '<p style="font-size:0.80rem;color:#6B8DB5;margin-top:0.4rem;">'
            '▶ Use <strong>Controls</strong> in the sidebar to start or stop monitoring.</p>',
            unsafe_allow_html=True,
        )

    # --- Bottom row: trend chart + recent records ---
    st.markdown("<div style='margin-top:1.4rem;'></div>", unsafe_allow_html=True)
    bot_left, bot_right = st.columns([3, 2], gap="large")

    with bot_left:
        st.markdown('<p class="section-label">📈&nbsp; Occupancy Trend</p>', unsafe_allow_html=True)
        st.markdown('<div class="bottom-card">', unsafe_allow_html=True)
        chart_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    with bot_right:
        st.markdown('<p class="section-label">📋&nbsp; Recent Records</p>', unsafe_allow_html=True)
        st.markdown('<div class="bottom-card">', unsafe_allow_html=True)
        log_table_placeholder    = st.empty()
        log_download_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Shared metric renderer ----
    def render_metrics(count: int, cap: int) -> None:
        pct, status, css = compute_occupancy(count, cap)
        available        = max(cap - count, 0)
        metric_status.markdown(_status_pill(status, css), unsafe_allow_html=True)
        metric_rows.markdown(
            _metric_row("👥 People Detected",   count)
            + _metric_row("🏫 Classroom Capacity", cap)
            + _metric_row("🪑 Available Seats",    available)
            + _metric_row("📊 Occupancy %",        f"{pct:.1f}%"),
            unsafe_allow_html=True,
        )
        progress_bar.markdown(_progress_bar(pct), unsafe_allow_html=True)
        for _ph in (metric_people, metric_capacity, metric_avail, metric_pct):
            _ph.markdown("", unsafe_allow_html=True)

    # ---- Shared log renderer ----
    def render_log_section() -> None:
        df_log = read_csv_tail(20)
        if df_log.empty:
            log_table_placeholder.info("No records yet. Start monitoring to begin logging.")
        else:
            log_table_placeholder.dataframe(df_log, use_container_width=True, hide_index=True)
        log_download_placeholder.download_button(
            label="⬇️ Download CSV",
            data=csv_bytes(),
            file_name="occupancy_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ===========================================================================
    # WebRTC component — rendered unconditionally when Browser WebRTC is
    # selected, regardless of monitoring state.
    #
    # WHY: desired_playing_state=False must reach the browser component while
    # it is still mounted in the DOM.  If the component is only rendered inside
    # the `else: # active monitoring` block it disappears from the DOM the
    # moment monitoring stops, so the explicit stop prop never arrives.
    # Rendering it here (always, for the WebRTC source) means:
    #   • monitoring=False  → component rendered, desired_playing_state=False
    #                         → browser receives stop signal → camera released.
    #   • monitoring=True   → component rendered, desired_playing_state=True
    #                         → browser starts camera automatically.
    #   • source = OpenCV   → component is NOT rendered at all (guarded below).
    # ===========================================================================
    _webrtc_ctx = None   # default; set only when WebRTC source is active

    if camera_source == "Browser Webcam (WebRTC)" or st.session_state.webrtc_stopping:
        _RTC_CONFIG = RTCConfiguration(
            iceServers=[{"urls": ["stun:stun.l.google.com:19302",
                                  "stun:stun1.l.google.com:19302"]}]
        )

        # Detector must exist before the processor factory runs.
        # Load it now (outside the monitoring gate) so the factory closure
        # always has a valid reference.  If the model is not yet loaded and
        # monitoring is off, skip rendering the component.
        if st.session_state.detector is None or st.session_state.detector_conf != confidence_threshold:
            if st.session_state.monitoring:
                # Only block-load the model when the user has actually started
                # monitoring; otherwise leave _webrtc_ctx as None.
                with st.spinner("⏳ Loading YOLO model…"):
                    try:
                        st.session_state.detector      = PersonDetector(
                            model_path="yolo11n.pt",
                            confidence=confidence_threshold,
                        )
                        st.session_state.detector_conf = confidence_threshold
                    except Exception as exc:
                        st.error(f"❌ Model load failed: {exc}")
                        st.session_state.monitoring = False

        # Only render the component when a detector is available (or monitoring
        # is off and we want desired_playing_state=False to reach a previously
        # live component).
        _det_for_webrtc = st.session_state.detector

        if _det_for_webrtc is not None:
            _det_ref = _det_for_webrtc  # local alias captured by closure

            def _make_processor() -> ClassroomVideoProcessor:
                return ClassroomVideoProcessor(_det_ref)

            with video_placeholder:
                # Render inside the left-column video placeholder so it
                # appears where the feed belongs in the layout.
                _webrtc_ctx = webrtc_streamer(
                    key="classroom_webrtc",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=_RTC_CONFIG,
                    video_processor_factory=_make_processor,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                    # This prop is forwarded to the browser component on every
                    # render.  When monitoring stops, the component is still
                    # mounted and receives False → browser stops the camera
                    # and closes the peer connection cleanly.
                    desired_playing_state=st.session_state.monitoring,
                )

            # ── Stop-transition guard ──────────────────────────────────────
            # During a source switch (webrtc_stopping=True), keep the component
            # rendered until the browser confirms it has stopped.
            # state.playing=False arrives synchronously with track.stop() (at
            # STOPPING dispatch); state.signalling=False confirms no offer pending.
            # This is the only Python-observable completion signal in this library.
            if (st.session_state.webrtc_stopping
                    and _webrtc_ctx is not None
                    and not _webrtc_ctx.state.playing
                    and not _webrtc_ctx.state.signalling):
                st.session_state.webrtc_stopping = False
                st.rerun()

        else:
            # ── Detector-absent escape hatch ───────────────────────────────
            # webrtc_stopping=True but the detector was never loaded, meaning
            # the WebRTC component was never rendered and the browser camera
            # was never opened.  There is nothing to wait for — clear the flag
            # immediately so the app does not loop indefinitely.
            if st.session_state.webrtc_stopping:
                st.session_state.webrtc_stopping = False
                st.rerun()

    # ---- IDLE state (OpenCV source, or WebRTC source but not yet monitoring) ----
    if not st.session_state.monitoring:
        # For OpenCV: show the idle placeholder inside the video column.
        # For WebRTC: the component above is already rendered in video_placeholder;
        #             it shows its own idle/stopped UI driven by desired_playing_state=False.
        #             Only fill the placeholder for OpenCV source.
        if camera_source == "Local Webcam (OpenCV)":
            video_placeholder.markdown("""
            <div class="video-idle">
                <div class="video-idle-icon">📷</div>
                <div class="video-idle-title">Webcam is Off</div>
                <div class="video-idle-sub">
                    Click <strong>▶ Start Monitoring</strong> in the sidebar to begin
                </div>
            </div>""", unsafe_allow_html=True)

        count_bar_placeholder.markdown(
            '<div class="people-count-bar">👥 0 people detected</div>',
            unsafe_allow_html=True,
        )
        metric_status.markdown(_status_pill("⚫ Monitoring Off", "status-off"), unsafe_allow_html=True)
        metric_rows.markdown(
            _metric_row("👥 People Detected",   0)
            + _metric_row("🏫 Classroom Capacity", int(capacity))
            + _metric_row("🪑 Available Seats",    int(capacity))
            + _metric_row("📊 Occupancy %",        "0.0%"),
            unsafe_allow_html=True,
        )
        progress_bar.markdown(_progress_bar(0), unsafe_allow_html=True)
        for _ph in (metric_people, metric_capacity, metric_avail, metric_pct):
            _ph.markdown("", unsafe_allow_html=True)

        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            chart_placeholder.line_chart(df.set_index("Time")["People"])

        render_log_section()

    # ---- ACTIVE MONITORING — branches by camera source ----
    else:
        # ==============================================================
        # Shared: ensure the YOLO detector is loaded (both branches use it)
        # ==============================================================
        if (st.session_state.detector is None
                or st.session_state.detector_conf != confidence_threshold):
            with st.spinner("⏳ Loading YOLO model…"):
                try:
                    st.session_state.detector      = PersonDetector(
                        model_path="yolo11n.pt",
                        confidence=confidence_threshold,
                    )
                    st.session_state.detector_conf = confidence_threshold
                except Exception as exc:
                    st.error(f"❌ Model load failed: {exc}")
                    st.session_state.monitoring = False
                    st.stop()

        detector: PersonDetector = st.session_state.detector

        if not detector.is_ready:
            st.error(f"❌ Detector error: {detector.load_error}")
            st.session_state.monitoring = False
            st.stop()

        # ==============================================================
        # BRANCH A — LOCAL WEBCAM (OpenCV) — original logic, UNCHANGED
        # ==============================================================
        if camera_source == "Local Webcam (OpenCV)":
            cam: WebcamCapture | None = st.session_state.cam
            if cam is None or not cam.is_open:
                cam = WebcamCapture(camera_index=int(camera_index))
                if not cam.open():
                    st.error(f"❌ {cam.error}")
                    st.session_state.monitoring = False
                    st.stop()
                st.session_state.cam = cam

            ok, frame = cam.read_frame()
            if not ok:
                st.warning("⚠️ Could not read frame from webcam — retrying…")
                time.sleep(0.05)
                st.rerun()

            annotated, count, _ = detector.detect(frame)
            st.session_state.people_count  = count
            st.session_state.frame_count  += 1

            st.session_state.history.append({
                "Time":   pd.Timestamp.now().strftime("%H:%M:%S"),
                "People": count,
            })
            if len(st.session_state.history) > 60:
                st.session_state.history.pop(0)

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_placeholder.image(Image.fromarray(rgb), channels="RGB", use_container_width=True)

            count_bar_placeholder.markdown(
                f'<div class="people-count-bar">👥 {count} {"person" if count == 1 else "people"} detected</div>',
                unsafe_allow_html=True,
            )

            render_metrics(count, int(capacity))

            df = pd.DataFrame(st.session_state.history)
            chart_placeholder.line_chart(df.set_index("Time")["People"])

            if st.session_state.frame_count != st.session_state.last_logged_frame:
                pct_now, status_now, _ = compute_occupancy(count, int(capacity))
                log_to_csv(count, int(capacity), pct_now, status_now)
                st.session_state.last_logged_frame = st.session_state.frame_count

            render_log_section()

            st.sidebar.caption(f"🎞️ Frame #{st.session_state.frame_count} | 👥 {count} detected")

            time.sleep(0.05)
            st.rerun()

        # ==============================================================
        # BRANCH B — BROWSER WEBCAM (WebRTC)
        # ==============================================================
        # The webrtc_streamer component was already rendered above the
        # idle/active gate (so it stays mounted and receives
        # desired_playing_state=False when monitoring stops).
        # _webrtc_ctx is set there; use it directly in the fragment.
        else:
            # ------------------------------------------------------------------
            # Periodic dashboard refresh — @st.fragment(run_every=1) re-executes
            # only this inner block once per second without triggering a full-page
            # rerun. This keeps the main Streamlit thread unblocked and prevents
            # any conflict with the WebRTC event loop.
            # The fragment captures _webrtc_ctx, capacity, and all placeholders
            # from the enclosing scope via closure — no arguments needed.
            # ------------------------------------------------------------------
            @st.fragment(run_every=1)
            def _webrtc_dashboard_fragment():
                # ── Stream is live ──────────────────────────────────────────
                if (_webrtc_ctx
                        and _webrtc_ctx.state.playing
                        and _webrtc_ctx.video_processor):

                    _proc = _webrtc_ctx.video_processor

                    # Thread-safe snapshot from the background processor thread.
                    _count, _frames, _err = _proc.get_snapshot()

                    # ── Error handling ───────────────────────────────────────
                    # On inference error: show warning, keep the last valid count,
                    # do NOT append to history, do NOT log to CSV.
                    if _err:
                        st.warning(f"⚠️ Detection error: {_err}")
                        # Restore the last successfully detected count so the
                        # dashboard does not incorrectly show zero.
                        _display_count = st.session_state.webrtc_last_valid_count
                    else:
                        _display_count = _count
                        # Persist the good count for future error recovery.
                        st.session_state.webrtc_last_valid_count = _count

                    # ── Duplicate-frame guard ────────────────────────────────
                    # Only update history / CSV when the processor has emitted a
                    # frame we have not yet consumed (frame counter advanced).
                    _new_frame = (
                        _frames > st.session_state.webrtc_last_consumed_frame
                        and _frames > 0
                        and not _err          # never record a failed frame
                    )

                    if _new_frame:
                        # Update rolling session-state counters
                        st.session_state.people_count  = _display_count
                        st.session_state.frame_count   = _frames
                        st.session_state.webrtc_last_consumed_frame = _frames

                        # Append to rolling 60-entry history
                        st.session_state.history.append({
                            "Time":   pd.Timestamp.now().strftime("%H:%M:%S"),
                            "People": _display_count,
                        })
                        if len(st.session_state.history) > 60:
                            st.session_state.history.pop(0)

                        # CSV — only from the main Streamlit thread (here), never
                        # from the WebRTC background callback.
                        if _frames != st.session_state.last_logged_frame:
                            pct_now, status_now, _ = compute_occupancy(
                                _display_count, int(capacity)
                            )
                            log_to_csv(_display_count, int(capacity), pct_now, status_now)
                            st.session_state.last_logged_frame = _frames

                    # ── UI updates (every fragment run, not just new frames) ──
                    # Always reflect the best available count even if no new frame.
                    _ui_count = st.session_state.people_count

                    count_bar_placeholder.markdown(
                        f'<div class="people-count-bar">👥 {_ui_count} '
                        f'{"person" if _ui_count == 1 else "people"} detected</div>',
                        unsafe_allow_html=True,
                    )

                    render_metrics(_ui_count, int(capacity))

                    if st.session_state.history:
                        df = pd.DataFrame(st.session_state.history)
                        chart_placeholder.line_chart(df.set_index("Time")["People"])

                    st.sidebar.caption(
                        f"🎞️ Frame #{_frames} | 👥 {_ui_count} detected (WebRTC)"
                    )

                # ── Stream not yet playing / stopped ────────────────────────
                else:
                    _status_label = (
                        "🌐 Waiting for browser camera…"
                        if st.session_state.monitoring
                        else "⚫ Monitoring Off"
                    )

                    count_bar_placeholder.markdown(
                        '<div class="people-count-bar">👥 — awaiting stream</div>',
                        unsafe_allow_html=True,
                    )
                    metric_status.markdown(
                        _status_pill(_status_label, "status-off"),
                        unsafe_allow_html=True,
                    )
                    metric_rows.markdown(
                        _metric_row("👥 People Detected",    "—")
                        + _metric_row("🏫 Classroom Capacity", int(capacity))
                        + _metric_row("🪑 Available Seats",    int(capacity))
                        + _metric_row("📊 Occupancy %",        "—"),
                        unsafe_allow_html=True,
                    )
                    progress_bar.markdown(_progress_bar(0), unsafe_allow_html=True)
                    for _ph in (metric_people, metric_capacity, metric_avail, metric_pct):
                        _ph.markdown("", unsafe_allow_html=True)

                render_log_section()

            # Execute the fragment immediately on this rerun so placeholders are
            # filled even before the first periodic tick fires.
            _webrtc_dashboard_fragment()

# ---------------------------------------------------------------------------
# PAGE: ANALYTICS
# ---------------------------------------------------------------------------
elif _page == "Analytics":
    st.markdown('<p class="section-label">📈&nbsp; Occupancy Analytics</p>', unsafe_allow_html=True)

    df_all = read_csv_tail(200)   # read more rows for analytics

    if df_all.empty:
        st.info("📋 No data yet. Start monitoring on the Dashboard to collect occupancy data.")
    else:
        df_all["Occupancy %"] = pd.to_numeric(df_all["Occupancy %"], errors="coerce")
        df_all["People Count"] = pd.to_numeric(df_all["People Count"], errors="coerce")

        # --- Summary stats ---
        a1, a2, a3, a4 = st.columns(4, gap="medium")
        with a1:
            st.markdown(_top_card("📊", "Avg Occupancy",  f"{df_all['Occupancy %'].mean():.1f}", "%"),
                        unsafe_allow_html=True)
        with a2:
            st.markdown(_top_card("🔺", "Peak Occupancy", f"{df_all['Occupancy %'].max():.1f}", "%"),
                        unsafe_allow_html=True)
        with a3:
            st.markdown(_top_card("👥", "Avg People",     f"{df_all['People Count'].mean():.1f}"),
                        unsafe_allow_html=True)
        with a4:
            st.markdown(_top_card("🔢", "Total Records",  len(df_all)),
                        unsafe_allow_html=True)

        # --- Occupancy % over time chart ---
        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.80rem;font-weight:700;color:#1E3A5F;">Occupancy % Over Time</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="bottom-card">', unsafe_allow_html=True)
        st.line_chart(df_all.set_index("Timestamp")["Occupancy %"])
        st.markdown('</div>', unsafe_allow_html=True)

        # --- People count over time ---
        st.markdown(
            '<p style="font-size:0.80rem;font-weight:700;color:#1E3A5F;margin-top:1rem;">People Count Over Time</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="bottom-card">', unsafe_allow_html=True)
        st.line_chart(df_all.set_index("Timestamp")["People Count"])
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PAGE: HISTORY
# ---------------------------------------------------------------------------
elif _page == "History":
    st.markdown('<p class="section-label">📋&nbsp; Occupancy History</p>', unsafe_allow_html=True)

    df_hist = read_csv_tail(200)
    st.markdown('<div class="bottom-card">', unsafe_allow_html=True)

    if df_hist.empty:
        st.info("No records yet. Start monitoring on the Dashboard to begin logging.")
    else:
        st.markdown(
            f'<p style="font-size:0.80rem;color:#6B8DB5;margin-bottom:0.5rem;">'
            f'Showing {len(df_hist)} most recent records (newest first)</p>',
            unsafe_allow_html=True,
        )
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

    _dl_col, _sp_col = st.columns([1, 3])
    with _dl_col:
        st.download_button(
            label="⬇️ Download Full CSV",
            data=csv_bytes(),
            file_name="occupancy_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PAGE: SETTINGS
# ---------------------------------------------------------------------------
elif _page == "Settings":
    st.markdown('<p class="section-label">⚙️&nbsp; Settings</p>', unsafe_allow_html=True)
    st.markdown('<div class="bottom-card">', unsafe_allow_html=True)
    st.markdown(
        '<p class="status-card-title">System Configuration</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        _metric_row("Classroom Capacity",     f"{int(capacity)} students")
        + _metric_row("Detection Confidence", f"{confidence_threshold:.2f}")
        + _metric_row("Camera Index",         int(camera_index))
        + _metric_row("YOLO Model",           "yolo11n.pt")
        + _metric_row("CSV Log File",         "occupancy_history.csv")
        + _metric_row("App Version",          "v1.0"),
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.info(
        "💡 **Tip:** Adjust Classroom Capacity, Detection Confidence, and Camera Index "
        "in the sidebar. Changes take effect immediately on the next monitoring frame."
    )
    st.markdown('</div>', unsafe_allow_html=True)
