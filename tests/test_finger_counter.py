import unittest

import cv2
import numpy as np

from finger_counter import detect_fingers


class FingerCounterTests(unittest.TestCase):
    def test_returns_zero_for_empty_image(self):
        image = np.zeros((320, 320, 3), dtype=np.uint8)
        result = detect_fingers(image)
        self.assertEqual(result.count, 0)
        self.assertEqual(result.boxes, [])

    def test_detects_five_synthetic_fingers(self):
        image = np.zeros((320, 320, 3), dtype=np.uint8)
        cv2.rectangle(image, (90, 140), (230, 290), (255, 255, 255), thickness=-1)

        for x in (95, 125, 155, 185, 215):
            cv2.rectangle(image, (x - 10, 60), (x + 10, 140), (255, 255, 255), thickness=-1)

        result = detect_fingers(image)

        self.assertEqual(result.count, 5)
        self.assertEqual(len(result.boxes), 5)


if __name__ == "__main__":
    unittest.main()
