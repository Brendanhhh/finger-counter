from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class FingerDetectionResult:
    count: int
    boxes: List[Tuple[int, int, int, int]]


def _build_hand_mask(image: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(image, (5, 5), 0)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 30, 30], dtype=np.uint8)
    upper_hsv = np.array([35, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    ycrcb = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb)
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if cv2.countNonZero(skin_mask) < 500:
        combined_mask = otsu_mask
    else:
        combined_mask = cv2.bitwise_or(skin_mask, otsu_mask)

    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return combined_mask


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _dedupe_points(points: Sequence[Tuple[int, int]], min_distance: int = 20) -> List[Tuple[int, int]]:
    unique: List[Tuple[int, int]] = []
    for point in sorted(points, key=lambda p: p[0]):
        if all((point[0] - ux) ** 2 + (point[1] - uy) ** 2 >= min_distance**2 for ux, uy in unique):
            unique.append(point)
    return unique


def _contour_profile_fingertips(mask: np.ndarray, contour: np.ndarray, cy: int) -> List[Tuple[int, int]]:
    x, y, w, h = cv2.boundingRect(contour)
    roi = mask[y : y + h, x : x + w]
    tops = np.full(w, -1, dtype=np.int32)

    for col in range(w):
        rows = np.where(roi[:, col] > 0)[0]
        if rows.size > 0:
            tops[col] = y + int(rows.min())

    valid_cols = np.where(tops >= 0)[0]
    if valid_cols.size < 3:
        return []

    profile = tops.astype(np.float32)
    missing_mask = profile < 0
    profile[missing_mask] = np.interp(
        np.where(missing_mask)[0], valid_cols, profile[valid_cols], left=profile[valid_cols[0]], right=profile[valid_cols[-1]]
    )
    profile = np.convolve(profile, np.ones(7, dtype=np.float32) / 7.0, mode="same")

    peaks: List[Tuple[int, int]] = []
    for i in range(1, w - 1):
        if profile[i] < profile[i - 1] and profile[i] <= profile[i + 1] and profile[i] < cy:
            peaks.append((x + i, int(profile[i])))

    return _dedupe_points(peaks, min_distance=max(15, w // 14))


def detect_fingers(image: np.ndarray) -> FingerDetectionResult:
    mask = _build_hand_mask(image)
    contour = _largest_contour(mask)

    if contour is None or cv2.contourArea(contour) < 1000:
        return FingerDetectionResult(count=0, boxes=[])

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return FingerDetectionResult(count=0, boxes=[])

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 3:
        return FingerDetectionResult(count=0, boxes=[])

    defects = cv2.convexityDefects(contour, hull_indices)
    fingertip_candidates: List[Tuple[int, int]] = []

    if defects is not None:
        for i in range(defects.shape[0]):
            start_idx, end_idx, far_idx, depth = defects[i, 0]
            start = tuple(contour[start_idx][0])
            end = tuple(contour[end_idx][0])
            far = tuple(contour[far_idx][0])

            a = np.linalg.norm(np.array(end) - np.array(start))
            b = np.linalg.norm(np.array(far) - np.array(start))
            c = np.linalg.norm(np.array(end) - np.array(far))
            if b == 0 or c == 0:
                continue

            cosine_angle = (b * b + c * c - a * a) / (2 * b * c)
            cosine_angle = max(-1.0, min(1.0, float(cosine_angle)))
            angle = np.degrees(np.arccos(cosine_angle))

            if angle < 85 and depth > 1000:
                fingertip_candidates.extend([start, end])

    fingertip_candidates.extend(_contour_profile_fingertips(mask, contour, cy))

    if not fingertip_candidates:
        hull_points = cv2.convexHull(contour, returnPoints=True)
        fingertip_candidates = [tuple(pt[0]) for pt in hull_points]

    unique_points = _dedupe_points(fingertip_candidates, min_distance=25)
    area = cv2.contourArea(contour)
    palm_radius = max(1.0, np.sqrt(area / np.pi) * 0.45)
    fingertip_points = []
    for x, y in unique_points:
        distance = float(np.hypot(x - cx, y - cy))
        if y < cy and distance > palm_radius:
            fingertip_points.append((x, y))

    if not fingertip_points:
        return FingerDetectionResult(count=0, boxes=[])

    h, w = image.shape[:2]
    box_size = max(20, min(h, w) // 12)
    boxes: List[Tuple[int, int, int, int]] = []

    for x, y in fingertip_points:
        x1 = max(0, x - box_size // 2)
        y1 = max(0, y - box_size // 2)
        x2 = min(w - 1, x + box_size // 2)
        y2 = min(h - 1, y + box_size // 2)
        boxes.append((x1, y1, x2, y2))

    boxes = sorted(boxes, key=lambda b: b[0])
    return FingerDetectionResult(count=len(boxes), boxes=boxes)


def annotate_fingers(image: np.ndarray, detection: FingerDetectionResult) -> np.ndarray:
    output = image.copy()
    for x1, y1, x2, y2 in detection.boxes:
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        output,
        f"Fingers: {detection.count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return output


def process_image(input_image_path: Path, output_image_path: Path) -> FingerDetectionResult:
    image = cv2.imread(str(input_image_path))
    if image is None:
        raise ValueError(f"Unable to read input image: {input_image_path}")

    detection = detect_fingers(image)
    annotated = annotate_fingers(image, detection)

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_image_path), annotated):
        raise ValueError(f"Unable to write output image: {output_image_path}")

    return detection


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect fingers in an image and draw boxes around them.")
    parser.add_argument("input_image", type=Path, help="Path to input image")
    parser.add_argument("output_image", type=Path, help="Path to write output image")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    detection = process_image(args.input_image, args.output_image)
    print(f"Detected fingers: {detection.count}")
    print(f"Saved annotated image to: {args.output_image}")


if __name__ == "__main__":
    main()
