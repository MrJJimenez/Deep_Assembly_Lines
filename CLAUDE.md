# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-camera 3D scene visualization platform for monitoring battery and screw assembly processes. Built for the TUM Practical Laboratory: Human Activity Understanding. Combines YOLOv11 (segmentation), DOPE (6D pose estimation), VGGT (3D reconstruction), and LSTM/Transformer models (sequence error detection) with a web-based 3D interface.

## Setup & Run

```bash
# Environment setup
conda create -n HAUP python=3.10 -y
conda activate HAUP

# PyTorch - macOS Apple Silicon
conda install pytorch::pytorch torchvision torchaudio -c pytorch -y
# PyTorch - NVIDIA GPU (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Dependencies
pip install -r requirements.txt

# Run the main application
python 3d_scene/3dscene.py
# Access at http://localhost:8085
```

There is no formal build system, test suite, or linter configured.

## Architecture

### Main Application (`3d_scene/`)
- **3dscene.py**: aiohttp async server (port 8085) — orchestrates multi-camera synchronized video playback, inference pipelines, and WebSocket communication with the frontend
- **web_interface.html**: Three.js-based 3D visualization frontend — renders camera frustums, object poses, point clouds, and assembly state overlays
- **yolo_inference.py**: YOLOv11 instance segmentation (6 classes: person, case, case_top, battery, screw, tool)
- **dope_inference.py**: DOPE 6D pose estimation for screwdriver and battery case (FP16 optimized)
- **vggt_inference.py**: 3D point cloud reconstruction from 7 cameras
- **lstm_inference.py**: Real-time error detection with multi-camera fusion (cameras 137322071489, 141722071426, 142122070087)

### State Machines (`3d_scene/`)
- **battery_fsm_module.py**: Tracks battery insertion sequence across 6 slots (expected order: 1→6). Uses SlotFSM (per-slot: IDLE→ENTERING→INSERTING→INSERTED) and OrderFSM (sequence validation). IoU threshold: 0.35, progress threshold: 0.55.
- **screw_sequence_tracker.py**: Tracks diagonal screw tightening pattern (BL→TR→BR→TL). States: IDLE→APPROACHING→SCREWING→COMPLETED.

### Training Module (`battery_order/`)
LSTM and Transformer models for assembly error detection with multiple cross-validation strategies (LOSO, LOCO, mixed). Key files: `model.py` (FrameLSTM), `feature_extraction.py`, `dataset.py`, training scripts (`lstm_*_cross-val.py`).

### YOLOv11 Fine-tuning (`yolov11_finetuned/`)
Training notebooks (Google Colab / A100 recommended) and inference scripts. Trained weights at `runs/segment/yolov11n_seg_custom/weights/best.pt`.

### External Frameworks (`frameworks/`)
- **dope/**: DOPE implementation (cuboid detection, PnP solving)
- **vggt/**: Visual Geometry Grounded Transformer (multi-view 3D reconstruction)

## Key Data

- **data/cams_calibrations.yml**: 8-camera extrinsics/intrinsics/distortion
- **data/recording_*/**: Synchronized multi-camera MP4 recordings (8 cameras each)
- **data/scanned_objects/**: 3D models (.obj/.mtl) for case and e-screwdriver
- **weights/**: DOPE weights (dope_tool.pth, dope_case.pth), VGGT weights (vggt.pt ~5GB), YOLOv11 weights

## Coordinate Systems

Code frequently converts between OpenCV (X-right, Y-down, Z-forward) and Three.js (X-right, Y-up, Z-backward). Watch for transformation matrices when modifying 3D visualization or pose estimation code.

## Dependencies

Core: PyTorch >=2.0, Ultralytics (YOLOv11), OpenCV >=4.8, NumPy <2.0, aiohttp >=3.9, aiortc >=1.6, scipy, pyrr, lap (BoT-SORT tracking). Frontend: Three.js r165+.
