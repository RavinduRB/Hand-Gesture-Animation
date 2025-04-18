import unittest
from src.gesture_detector import GestureDetector

class TestGestureDetector(unittest.TestCase):

    def setUp(self):
        self.detector = GestureDetector()

    def test_one_finger_gesture(self):
        # Simulate input for one finger gesture
        finger_positions = [(100, 200)]
        gesture = self.detector.detect_gesture(finger_positions)
        self.assertEqual(gesture, "one_finger")

    def test_two_finger_gesture(self):
        # Simulate input for two fingers gesture
        finger_positions = [(100, 200), (150, 250)]
        gesture = self.detector.detect_gesture(finger_positions)
        self.assertEqual(gesture, "two_fingers")

    def test_no_finger_gesture(self):
        # Simulate input for no fingers
        finger_positions = []
        gesture = self.detector.detect_gesture(finger_positions)
        self.assertEqual(gesture, "no_gesture")

if __name__ == '__main__':
    unittest.main()