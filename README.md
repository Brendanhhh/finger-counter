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
```
# Responsible AI Usage
This project adheres to the following principles of responsible and ethical AI development:

Data Privacy: All image processing is performed locally on the host machine. No video data or biometric information is recorded, stored, or transmitted to external servers.

Transparency & Attribution: This documentation and the project's structural framework were generated with the assistance of AI to ensure clarity and professional standards in academic reporting.

Bias Awareness: Users should be aware that performance may vary based on environmental factors such as lighting, background complexity, and varying hand shapes or skin tones. The underlying models are pre-trained, and limitations in the training data may impact accuracy in specific conditions.

Safety & Limitations: This software is intended for educational and research purposes. It is not designed for—and should not be used in—critical safety systems, biometric authentication, or medical applications where high-stakes reliability is required.

License
Distributed under the MIT License.
