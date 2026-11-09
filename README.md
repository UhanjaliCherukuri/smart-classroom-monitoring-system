# 🎓 Smart Classroom Monitoring System

A real-time classroom occupancy tracker built with **Python**, **OpenCV**, **YOLO11n**, and **Streamlit** — featuring a professional Blue & White dashboard design.

---

## 📋 Features

| # | Feature | Detail |
|---|---|---|
| 1 | 🎥 **Live Webcam Feed** | OpenCV DirectShow capture; graceful error handling |
| 2 | 🧍 **Person Detection** | Pretrained YOLO11n — persons-only class filter, confidence slider |
| 3 | 🔢 **People Counter** | Live count updated every frame (~20 FPS) |
| 4 | 📊 **Occupancy Dashboard** | Configurable capacity; %, status pill, progress bar |
| 5 | 📈 **Occupancy Trend Chart** | Rolling 60-frame line chart |
| 6 | 📝 **CSV History Logging** | Auto-created `occupancy_history.csv`; one row per frame |
| 7 | ⬇️ **CSV Download** | One-click download from the dashboard |
| 8 | 🎨 **Professional UI** | Blue & White design system — hero banner, metric cards, status badges |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

Browser opens automatically at **http://localhost:8501**.

---

## 🗂️ Project Structure

```
cv smart classroom monitoring system/
├── app.py                  # Streamlit dashboard (main entry point)
├── detector.py             # YOLO11n PersonDetector class
├── camera.py               # OpenCV WebcamCapture wrapper
├── occupancy_history.csv   # Auto-created on first run
├── yolo11n.pt              # Downloaded automatically on first run (~6 MB)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## ⚙️ Configuration (Sidebar)

| Setting | Default | Range | Description |
|---|---|---|---|
| Classroom Capacity | 30 | 1–200 | Max students the room holds |
| Detection Confidence | 0.40 | 0.10–0.90 | Minimum YOLO score to count a person |
| Camera Index | 0 | 0–5 | 0 = built-in webcam; 1+ = external |

---

## 📊 Occupancy Status Logic

| Count | % of Capacity | Status | Colour |
|---|---|---|---|
| 0 | 0% | 🟢 Empty | Green |
| 1 → <80% cap | <80% | 🟢 Occupied | Green |
| 80–99% cap | 80–99% | 🟡 Almost Full | Amber |
| ≥ capacity | 100% | 🔴 Capacity Reached | Red |

---

## 📝 CSV Occupancy Log

A file `occupancy_history.csv` is created automatically in the project folder.

**Columns:** `Timestamp`, `People Count`, `Capacity`, `Occupancy %`, `Status`

**Sample:**
```
Timestamp,People Count,Capacity,Occupancy %,Status
2026-11-07 17:30:01,3,30,10.0,Occupied
2026-11-07 17:30:02,5,30,16.7,Occupied
2026-11-07 17:30:04,24,30,80.0,Almost Full
```

- Written using Python's built-in `csv` module — no extra packages
- One row per processed frame; duplicate-guard prevents extra rows on Streamlit reruns
- Full CSV downloadable via the **⬇️ Download CSV** button on the dashboard

---

## 🛠️ Dependencies

| Package | Version | Purpose |
|---|---|---|
| `opencv-python` | 5.0+ | Webcam capture & image processing |
| `ultralytics` | 8.4+ | YOLO11n pretrained model |
| `streamlit` | 1.67+ | Web dashboard UI |
| `pandas` | 3.0+ | History data & CSV display |
| `Pillow` | 13.0+ | Frame BGR→RGB conversion |
| `torch` | 2.15+ | YOLO inference backend |

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| Webcam not found | Change Camera Index in sidebar (try 1, then 2) |
| Model download slow on first run | Downloads ~6 MB once; cached after that |
| People not detected | Lower Confidence slider (try 0.25–0.35) |
| Too many false positives | Raise Confidence slider (try 0.55–0.65) |
| Stop button feels slow | Normal — Streamlit processes it on the next frame rerun |
| `occupancy_history.csv` missing | Starts the app once — file is auto-created |

---

## 🏗️ Architecture Note

The app uses a **one-frame-per-rerun** pattern to keep Streamlit's event loop responsive:

```
st.rerun()
   │
   ▼
 Read 1 webcam frame
   │
   ▼
 YOLO detection
   │
   ▼
 Update placeholders (video, metrics, chart, log)
   │
   ▼
 Write CSV row (if new frame)
   │
   ▼
time.sleep(0.05)  →  st.rerun()
```

This allows the **Stop button** to register between frames rather than being blocked by a `while` loop.

---

## ⚠️ Limitations

- Counts all visible people (no identity or face tracking)
- Single camera per session
- Performance depends on CPU speed — no GPU required, but GPU accelerates inference automatically if available

---

*Smart Classroom Monitoring System v1.0 — Built for educational purposes*
