# finger-counter

A minimal OCR-inspired finger counter for images.

## What it does

- Input: an image path
- Output: number of detected fingers and an output image with boxes drawn around each detected fingertip

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python finger_counter.py <input_image> <output_image>
```

Example output in the terminal:

```text
Detected fingers: 5
Saved annotated image to: output.png
```
