# Finger Counter

## 📌 Overview
DigitDetect is a computer vision project designed to detect, bound, and count human fingers in images. Given an input image containing people, the model identifies visible fingers, draws bounding boxes around each one, and outputs a total count along with confidence scores for each detection. 

## ✨ Features
* **Finger Detection:** Accurately identifies individual fingers in diverse images.
* **Bounding Boxes:** Draws clear, labeled bounding boxes around each detected finger.
* **Confidence Scores:** Displays a model confidence percentage for every bounding box.
* **Total Count:** Outputs the total number of fingers detected in the image.
* **Batch Processing:** (Optional) Can process single images or iterate through a folder of images.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Computer Vision:** OpenCV, MediaPipe Hands
* **Data Processing:** NumPy

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)[YourUsername]/[YourRepositoryName].git
   cd [YourRepositoryName]
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(This project currently uses `opencv-python`, `mediapipe`, and `numpy`.)*

## 💻 Usage

To run the finger counting model on a single image, use the following command:

```bash
python src/detect.py --source path/to/your/image.jpg
```

**Arguments:**
* `--source`: Path to the input image or directory containing images.
* `--conf`: (Optional) Confidence threshold for detections (default is 0.5).
* `--save`: (Optional) Flag to save the output image with bounding boxes drawn.

**Example Output Console:**
```text
Processing image.jpg...
Detected 8 fingers.
Confidence Scores: [0.98, 0.95, 0.96, 0.89, 0.92, 0.94, 0.91, 0.88]
Output saved to runs/detect/image_out.jpg
```

## 📂 Project Structure
```text
├── data/                   # Sample images for testing
├── models/                 # Pre-trained weights and model architecture files
├── src/                    # Source code for data processing and inference
│   ├── detect.py           # Main inference script
│   └── utils.py            # Helper functions for drawing boxes and counting
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 🚧 Challenges & Future Improvements
* **Occlusion & Overlap:** Fingers are often hidden behind other objects or hands. Improving the model's accuracy on partially visible fingers.
* **Real-time Video Support:** Extending the script to run live inference via a webcam feed.
* **Hand Orientation:** Categorizing left vs. right hands alongside finger detection.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome. Feel free to check the issues page.

## 📄 License
Distributed under the MIT License. See LICENSE for more information.
