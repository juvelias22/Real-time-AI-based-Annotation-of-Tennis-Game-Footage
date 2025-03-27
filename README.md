# 🎾 Real-Time AI-Based Annotation of Tennis Game Footage

This repository provides the implementation of a real-time, AI-powered framework for automatic annotation and commentary of singles tennis matches. The system integrates multiple computer vision and deep learning models to detect players, track the ball, classify strokes, segment rallies, and generate natural language commentary using structured match annotations.

> 🧠 Developed as part of a Master's dissertation on artificial intelligence.

---

## 📦 Features

- ⚡ Real-time inference with multi-threaded processing
- 🧍 Player detection and filtering via YOLOv8
- 🎯 Ball tracking using TrackNet
- 🎾 Pose-based action recognition (Forehand, Backhand)
- 🧠 Commentary generation using OpenAI GPT-3.5
- 📊 Rally segmentation
- 📌 Tennis court keypoint detection using ResNet18
- 🧾 Scoreboard text recognition via EasyOCR


## 🧠 System Overview

The framework includes:

- **YOLOv8** for player and net detection  
- **YOLOv8-Pose** for joint keypoint estimation  
- **TrackNet** for ball tracking  
- **ResNet-18** for tennis court keypoint detection  
- **Scikit-learn** model for forehand/backhand classification  
- **Rally segmentation** using a binary classifier  
- **OCR** using EasyOCR for scoreboard overlays  
- **Custom algorithm** for ball hit detection  
- **ChatGPT-3.5 Turbo** for commentary generation

All components are orchestrated to process standard broadcast footage from a single camera angle and generate live annotations and fluent, contextual commentary.

📦 Download all pretrained model weights (YOLOv8, TrackNet, ResNet18, etc.) from the following Google Drive link:
https://drive.google.com/drive/folders/1PtZ8_7CEdhP0sI9bhwt8BDwx91rPe4Vj?usp=drive_link
