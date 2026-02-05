# Human Activity Understanding

This repository contains tools for multi-camera 3D scene reconstruction, object pose estimation, object detection, and a screw-sequence tracking demo that leverages a dedicated distance-tool frontend.
Key components
 - `3d_scene/`: main demo and server (synchronized multi-camera playback, DOPE + YOLO integration, web frontend).
 - `frameworks/dope/`: DOPE 6D pose estimation code and model hooks.
What this README covers
 - Quickstart: how to run the demo and the CLI monitor
 - DOPE tuning & performance tips (Apple Silicon / MPS guidance)
**Requirements**
- Python 3.10+ (Apple Silicon users: prefer the system Python from macOS or a compatible installer that supports MPS)
- PyTorch with MPS support for Apple Silicon (or CUDA on Linux/Windows if applicable)
**Install (example)**
1. Create and activate a virtual environment:
```bash
2. Install dependencies. There is no single global `requirements.txt`; install the requirements for the frameworks you intend to run. For example, for the YOLO experiments:
```bash
pip install -r yolov11_finetuned/requirements.txt
Quickstart — run the backend server and open the frontend

1. Start the backend server (serves the visualization and provides REST endpoints):
```bash
2. Open the web frontend in your browser:
- Visit: `http://localhost:8085/` (or the address printed by the server log)
3. Optional: run the CLI monitor (shows screw-sequence progress in terminal):
```bash
python 3d_scene/sequence_from_distance_tool.py
Screw-sequence tracker (what it does)
- Purpose: track a screw insertion/removal sequence using distance readings from the frontend tool-case UI and a small state machine.
- Implementation: `3d_scene/screw_sequence_tracker.py` contains `ScrewSequenceTracker` and the thresholds used for approach/screwing detection.
Key screw-tracker endpoints
- POST `/api/distance/tool-case` — payload: `{"distance_cm": <float>, "nearest_screw": "top_left"|"top_right"|"bottom_left"|"bottom_right"}`
- GET `/api/screw/status` — returns tracker state, current sequence and per-screw progress
Example: sending distance updates via curl
```bash
curl -X POST http://localhost:8085/api/distance/tool-case \
DOPE tuning & performance tips
- Location of main runtime flags: `3d_scene/3dscene.py` exposes DOPE flags such as `DOPE_ENABLED`, `DOPE_USE_FP16`, and `DOPE_STOP_AT_STAGE`.
- Config file: `3d_scene/config/config_pose.yaml` contains `downscale_height` — increasing this improves detection detail at the cost of FPS.
- Inference frequency: `SyncedVideoManager.dope_inference_interval` controls how often DOPE runs (every N frames). You can change it on the fly via the API:
  - POST `/api/dope/interval` with `{"interval": <int>}` to set the interval.
Example to set DOPE to run every 3 frames:
Recommended trade-offs (Apple Silicon / MPS)
- If you need higher FPS: use `DOPE_USE_FP16=True`, reduce `downscale_height` (e.g., 240), and increase `dope_inference_interval`.
- If you need highest detection robustness: set `DOPE_STOP_AT_STAGE` to a higher stage (up to 6), set `DOPE_USE_FP16=False`, and use a larger `downscale_height` (e.g., 400 or higher). This reduces FPS but improves accuracy.
Recordings and dataset notes
- The `data/` folder contains `recording_1`, `recording_2`, … `recording_12` with camera MP4s and calibration entries. The demo was tested with 12 recordings; individual recordings contain the same set of camera IDs used by the projector / viewer.
- Change the default recording used by the demo by editing `RECORDING_DIR` in `3d_scene/3dscene.py`.
Developer notes & important files
- Server entry: `3d_scene/3dscene.py` — main server, video sync, DOPE and YOLO integration, REST endpoints.
- Frontend: `3d_scene/web_interface.html` — Three.js visualization, distance tool UI, and code that posts distance information to the backend. The vertex→screw mapping lives here and must match camera orientation.
- Tracker: `3d_scene/screw_sequence_tracker.py` — state machine thresholds can be tuned there: `APPROACH_DISTANCE`, `SCREWING_DISTANCE`, `FRAMES_TO_COMPLETE`.
- DOPE detector: `frameworks/dope/detector.py` — model loading and inference. Avoid enabling `torch.compile()` on MPS unless validated on your setup.
Troubleshooting
- Frontend shows `Dist: N/A`: likely because DOPE is disabled or not loaded. Ensure `DOPE_ENABLED=True` in `3d_scene/3dscene.py` and check server logs for DOPE model loading messages.
- Detection quality poor: increase `downscale_height`, increase `DOPE_STOP_AT_STAGE`, disable FP16 if unstable.
- Frontend not loading: look for JavaScript errors in the browser console. A common regression was a removed DOM element while code still tried to access it; ensure the frontend code is consistent with the HTML.
- DOPE inference crashes on MPS after `torch.compile()`: revert the `torch.compile()` usage in `frameworks/dope/detector.py` (this repo previously reverted that change).
Testing & monitoring
- CLI monitor: `python 3d_scene/sequence_from_distance_tool.py` polls `/api/screw/status` and shows the screw-sequence progress in the terminal.

Next steps and customization ideas
- Expose screw-tracker thresholds in the REST API or frontend for easier calibration.
- Add a lightweight test harness that replays prerecorded distance traces against the tracker for automated validation.
Credits & license
- This project bundles multiple academic / research tools (DOPE, VGGT, YOLO variants). Respect each framework's license and project's model weight licenses in `weights/`.

If you want, I can further:
- Add a short script to set recommended DOPE settings for accuracy or speed.
- Expose tracker thresholds via REST endpoints and a small calibration UI.

---
Last updated: Feb 5, 2026
# Human Activity Understanding - Screw Assembly Tracking

This repository contains the final project for the **Human Activity Understanding** lab at TUM. The project implements a 3D scene visualization system that tracks a screw assembly task, detecting when an operator tightens screws on a case in the correct sequence.

## 🎯 Project Overview

The system uses multi-camera recordings to:

1. **Detect Objects in 3D** - Uses DOPE (Deep Object Pose Estimation) to detect the 6D pose of:
   - Electric screwdriver (tool)
   - Case with 4 screw positions

2. **Track Screw Sequence** - Monitors which screw is being tightened based on tool-to-screw distance and validates the correct order:
   - Expected sequence: **BL → TR → BR → TL** (diagonal pattern)

3. **Visualize in 3D** - Real-time web-based 3D visualization using Three.js

4. **Segment Objects** - YOLOv11 instance segmentation for object detection
5. **Track Battery Events** - Detects battery presence, insertion/removal, and charge state using YOLO segmentation and a dedicated FSM (see `battery_fsm_module.py`).

## 📁 Project Structure

```
├── 3d_scene/                    # Main application
│   ├── 3dscene.py              # Backend server (aiohttp)
│   ├── web_interface.html      # 3D visualization frontend (Three.js)
│   ├── screw_sequence_tracker.py   # Screw sequence state machine
│   ├── sequence_from_distance_tool.py  # CLI monitoring tool
│   ├── distance_tool_screw.py  # Distance API client
│   ├── dope_inference.py       # DOPE 6D pose estimation
│   ├── yolo_inference.py       # YOLOv11 segmentation
│   ├── vggt_inference.py       # 3D point cloud reconstruction
│   ├── battery_fsm_module.py   # Battery tracking state machine (YOLO-based)
│   └── config/                 # Camera calibrations & DOPE config
│
├── data/
│   ├── recording_1-12/         # Multi-camera recordings (8 cameras each)
│   ├── scanned_objects/        # 3D models (case, e-screwdriver)
│   └── cams_calibrations.yml   # Camera calibration data
│
├── weights/                    # Model weights
│   ├── dope_tool.pth          # DOPE weights for screwdriver
│   ├── dope_case.pth          # DOPE weights for case
│   └── model.pt               # YOLOv11 finetuned weights
│
├── frameworks/                 # External frameworks
│   ├── dope/                  # DOPE implementation
│   └── vggt/                  # VGGT point cloud
│
└── yolov11_finetuned/         # YOLOv11 training & testing
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision  # PyTorch with CUDA/MPS support
pip install aiohttp opencv-python numpy pyyaml requests
pip install ultralytics  # For YOLOv11
```

### 2. Run the 3D Scene Server

```bash
python 3d_scene/3dscene.py
```

This starts the backend server on `http://localhost:8085`.

### 3. Open the Web Interface

Open your browser and navigate to:
```
http://localhost:8085
```

You'll see the 3D visualization with:
- Multi-camera video feeds
- 3D models of tool and case
- Real-time screw tracking status
- Distance measurements

### 4. Monitor Screw Sequence (Optional CLI)

In a separate terminal, run the monitoring script:

```bash
python 3d_scene/sequence_from_distance_tool.py
```

This displays a real-time ASCII diagram of the screw tracking:

```
  ┌─────────────────┐
  │  ○ TL     TR ●  │
  │                 │
  │      CASE       │
  │                 │
  │  ✓ BL     BR ○  │
  └─────────────────┘
  State: SCREWING | Step: 2/4 | Active: TR | Dist: 3.2cm
  Progress: BL ✓
```

### 5. Monitor Battery Events (Optional)

Battery tracking is available via the backend API and YOLO-based tracking scripts. You can:
- Query battery status via `/api/battery/status`
- Simulate events via `/api/battery/event`
- Use YOLO scripts (e.g., `track_yolov11n-seg.py`) to visualize battery, screw, case, tool, and person tracks.

## ⚙️ Configuration

### Change Recording

Edit `3d_scene/3dscene.py` line 31:

```python
RECORDING_DIR = "data/recording_3"  # Change to any recording_1 through recording_12
```

### Adjust DOPE Settings

For faster inference (lower accuracy):
```python
DOPE_USE_FP16 = True      # Half-precision inference
DOPE_STOP_AT_STAGE = 1    # Stop at first stage
```

### Adjust Screw Tracking Thresholds

In `3d_scene/screw_sequence_tracker.py`:
```python
APPROACH_DISTANCE = 8.0   # cm - Start tracking when tool approaches
SCREWING_DISTANCE = 4.0   # cm - Consider "screwing" when this close
FRAMES_TO_COMPLETE = 40   # Frames at screwing distance = screw done
```

## 📊 Available Recordings

All 12 recordings are fully supported with 8 synchronized cameras each:

| Recording | Cameras | Status |
|-----------|---------|--------|
| recording_1 | 8 | ✅ Supported |
| recording_2 | 8 | ✅ Supported |
| recording_3 | 8 | ✅ Supported (default) |
| recording_4 | 8 | ✅ Supported |
| recording_5 | 8 | ✅ Supported |
| recording_6 | 8 | ✅ Supported |
| recording_7 | 8 | ✅ Supported |
| recording_8 | 8 | ✅ Supported |
| recording_9 | 8 | ✅ Supported |
| recording_10 | 8 | ✅ Supported |
| recording_11 | 8 | ✅ Supported |
| recording_12 | 8 | ✅ Supported |

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/screw/status` | GET | Get screw tracking status |
| `/api/screw/reset` | POST | Reset screw sequence |
| `/api/distance/tool-case` | GET | Get tool-to-case distance |
| `/api/dope/interval` | POST | Set DOPE inference interval |

**Battery Detection**
- **Purpose:** Detects the presence and approximate charge state of a rechargeable battery used in the tool/case (if present), and reports battery-related events alongside the screw-sequence tracking.
- **How it works:** The system uses a combination of visual cues (YOLO segmentation + DOPE pose) to determine whether a battery is present and whether it is being inserted.
- **Endpoints:**
  - **GET** `/api/battery/status` — returns JSON: `{"present": true|false, "level_pct": <0-100|null>, "last_event": "inserted"|"removed"|"unknown"}`.
  - **POST** `/api/battery/event` — accept manual/ground-truth events for testing: `{"event": "inserted"|"removed", "timestamp": <iso8601>}`.
- **Frontend UI:** The web interface will show a small battery icon next to the tool/case HUD with:
  - solid color when present and level known (green/yellow/red),
  - gray when unknown, and
  - flash when an insert/remove event is detected.
- **Integration with screw tracker:** If a battery removal/insertion event occurs during an assembly step, the tracker will annotate that step with the battery event in `/api/screw/status` so you can see temporal correlation between battery actions and screw progress.
- **Calibration & tuning:**
  - If battery level comes from visual estimation, tune YOLO segmentation thresholds to reliably segment the battery region (see `yolo_inference.py`).
  - If you have wired metadata from the tool (serial/USB/BLE), enable that input in `3d_scene/3dscene.py` and map the incoming signal to `/api/battery/event` for highest accuracy.
- **Testing:** Use the CLI or curl to simulate battery events:
  ```bash
  curl -X POST http://localhost:8085/api/battery/event -H "Content-Type: application/json" \
    -d '{"event":"removed","timestamp":"2026-02-05T12:34:56Z"}'
  ```
  Then `GET /api/battery/status` to verify the state.


## 🎥 Camera Setup

The system uses 8 synchronized Intel RealSense cameras:
- **Tool camera:** `142122070087` - Tracks screwdriver pose
- **Case camera:** `135122071615` - Tracks case pose
- **YOLO camera:** `137322071489` - Object segmentation
- **Additional cameras** for VGGT point cloud reconstruction

## 🛠️ Technologies

- **Backend:** Python, aiohttp, OpenCV
- **Frontend:** Three.js, WebGL
- **ML Models:**
  - DOPE - 6D object pose estimation
  - YOLOv11 - Instance segmentation
  - VGGT - 3D reconstruction
- **Hardware:** Intel RealSense cameras, MPS/CUDA GPU acceleration

## 📝 Expected Screw Sequence

The correct tightening order follows a diagonal pattern to ensure even pressure:

```
    1. BL (Bottom-Left)  - First corner
    2. TR (Top-Right)    - Diagonal opposite
    3. BR (Bottom-Right) - Adjacent to first
    4. TL (Top-Left)     - Final corner
```

The system tracks deviations and reports errors when the wrong screw is tightened.

## 🐛 Troubleshooting

### "Dist: N/A" in web interface
- Ensure `DOPE_ENABLED = True` in `3dscene.py`
- Check that both tool and case are visible in their respective camera views

### Low FPS on MacBook
- Increase `dope_inference_interval` (default: 15)
- Use the API: `curl -X POST http://localhost:8085/api/dope/interval -d '{"interval": 30}'`

### Wrong screw detected
- The vertex mapping may need adjustment for your camera setup
- Check console logs for `[Screw3D]` debug messages

---

**Course:** Practical Course - Human Activity Understanding (IN2106, IN4265)  
**Institution:** Technical University of Munich (TUM)
