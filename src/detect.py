import argparse
from pathlib import Path

import cv2
import mediapipe as mp

from utils import (
    collect_image_paths,
    draw_finger_boxes,
    ensure_output_dir,
    get_extended_finger_tips,
)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finger detection entry point.")
    parser.add_argument("--source", required=True, help="Image path or directory path.")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output images with visualization overlays.",
    )
    return parser.parse_args()


def run_inference(
    image_path: Path,
    conf_threshold: float,
    hands: mp_hands.Hands,
) -> tuple[int, list[float], any]:
    """Run MediaPipe Hands inference for one image and return count, confidences, and annotation."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    total_fingers = 0
    finger_scores: list[float] = []

    if not results.multi_hand_landmarks or not results.multi_handedness:
        return total_fingers, finger_scores, image

    for hand_landmarks, handedness in zip(
        results.multi_hand_landmarks,
        results.multi_handedness,
    ):
        hand_label = handedness.classification[0].label
        hand_score = float(handedness.classification[0].score)

        if hand_score < conf_threshold:
            continue

        landmarks = hand_landmarks.landmark
        extended_tips = get_extended_finger_tips(landmarks=landmarks, hand_label=hand_label)
        total_fingers += len(extended_tips)
        finger_scores.extend([round(hand_score, 3)] * len(extended_tips))

        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        draw_finger_boxes(image=image, landmarks=landmarks, tip_indices=extended_tips)

    cv2.putText(
        image,
        f"Finger Count: {total_fingers}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return total_fingers, finger_scores, image


def main() -> None:
    args = parse_args()
    image_paths = collect_image_paths(args.source)

    output_dir = ensure_output_dir() if args.save else None

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=4,
        min_detection_confidence=args.conf,
        min_tracking_confidence=args.conf,
    ) as hands:
        for image_path in image_paths:
            print(f"Processing {image_path.name}...")
            count, scores, annotated_image = run_inference(
                image_path=image_path,
                conf_threshold=args.conf,
                hands=hands,
            )
            print(f"Detected {count} fingers.")
            print(f"Confidence Scores: {scores}")

            if output_dir is not None:
                output_path = output_dir / image_path.name
                cv2.imwrite(str(output_path), annotated_image)
                print(f"Output saved to {output_path}")


if __name__ == "__main__":
    main()
