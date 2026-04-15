from pathlib import Path

import cv2


def collect_image_paths(source: str) -> list[Path]:
    """Return image files from a file path or a directory path."""
    source_path = Path(source)

    if source_path.is_file():
        return [source_path]

    if source_path.is_dir():
        patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        paths: list[Path] = []
        for pattern in patterns:
            paths.extend(source_path.glob(pattern))
        return sorted(paths)

    raise FileNotFoundError(f"Source not found: {source}")


def ensure_output_dir(base_dir: str = "runs/detect") -> Path:
    """Create and return the default output directory."""
    output_dir = Path(base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def is_thumb_extended(landmarks: list, hand_label: str) -> bool:
    """Determine whether the thumb is extended using handedness-aware x-position logic."""
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    if hand_label == "Right":
        return thumb_tip.x < thumb_ip.x
    return thumb_tip.x > thumb_ip.x


def is_finger_extended(landmarks: list, tip_idx: int, pip_idx: int) -> bool:
    """Determine whether a non-thumb finger is extended in the image plane."""
    return landmarks[tip_idx].y < landmarks[pip_idx].y


def get_extended_finger_tips(landmarks: list, hand_label: str) -> list[int]:
    """Return landmark indices for finger tips that appear extended."""
    extended_tips: list[int] = []

    if is_thumb_extended(landmarks, hand_label):
        extended_tips.append(4)

    finger_pairs = ((8, 6), (12, 10), (16, 14), (20, 18))
    for tip_idx, pip_idx in finger_pairs:
        if is_finger_extended(landmarks, tip_idx, pip_idx):
            extended_tips.append(tip_idx)

    return extended_tips


def draw_finger_boxes(image, landmarks: list, tip_indices: list[int]) -> None:
    """Draw fixed-size boxes around detected finger tips."""
    image_h, image_w = image.shape[:2]
    half_box_size = 16

    for tip_idx in tip_indices:
        tip = landmarks[tip_idx]
        x = int(tip.x * image_w)
        y = int(tip.y * image_h)

        x1 = max(0, x - half_box_size)
        y1 = max(0, y - half_box_size)
        x2 = min(image_w - 1, x + half_box_size)
        y2 = min(image_h - 1, y + half_box_size)
        cv2.rectangle(image, (x1, y1), (x2, y2), (40, 190, 255), 2)
