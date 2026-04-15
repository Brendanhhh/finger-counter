# Finger Counter AI

A computer vision project utilizing Python and machine learning to detect and count extended fingers in real-time. This project was developed for academic purposes to demonstrate hand landmark localization and gesture interpretation.

## Overview

This project leverages the **Mediapipe** framework and **OpenCV** to track hand movements via a webcam. By identifying specific coordinates (landmarks) on the hand, the algorithm determines the state of each finger (extended or folded) to provide an accurate count.

## Features

* **Real-Time Detection:** Low-latency processing for immediate feedback.
* **Landmark Visualization:** Overlays hand skeletons and connection points on the video feed.
* **Dual Hand Support:** Capable of recognizing and counting fingers on both left and right hands.
* **Dynamic Display:** Shows the current count and FPS (Frames Per Second) directly on the output window.

## Requirements

* Python 3.x
* OpenCV (`opencv-python`)
* Mediapipe

## Installation

```bash
pip install opencv-python mediapipe
