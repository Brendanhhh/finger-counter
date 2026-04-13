"""Simple webcam-based finger counting MVP."""

from __future__ import annotations

import cv2
import mediapipe as mp


TIP_IDS = (4, 8, 12, 16, 20)
PIP_IDS = (3, 6, 10, 14, 18)


def count_raised_fingers(hand_landmarks, handedness: str) -> int:
    landmarks = hand_landmarks.landmark
    raised = 0

    thumb_tip = landmarks[TIP_IDS[0]]
    thumb_joint = landmarks[PIP_IDS[0]]
    if handedness == "Right":
        raised += int(thumb_tip.x < thumb_joint.x)
    else:
        raised += int(thumb_tip.x > thumb_joint.x)

    for tip_id, pip_id in zip(TIP_IDS[1:], PIP_IDS[1:]):
        raised += int(landmarks[tip_id].y < landmarks[pip_id].y)

    return raised


def main() -> None:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Could not open webcam.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            finger_count = 0
            label = "Unknown"

            if results.multi_hand_landmarks and results.multi_handedness:
                hand_landmarks = results.multi_hand_landmarks[0]
                label = results.multi_handedness[0].classification[0].label
                finger_count = count_raised_fingers(hand_landmarks, label)

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

            cv2.putText(
                frame,
                f"Hand: {label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Fingers: {finger_count}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Finger Counter MVP", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
