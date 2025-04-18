# Hand Gesture Controlled Animation System 🚀

This interactive system uses real time hand gesture recognition to control engaging visual animations like bubbles and rain using just your fingers. Powered by MediaPipe and OpenCV, the system reads webcam input and tracks hand movements to switch between different animations, offering an intuitive and fun way to interact with your screen.




https://github.com/user-attachments/assets/2e82424a-d2e9-42b2-a989-80da96e15e32




### ❄️ Key Features
- Real Time Hand Gesture Detection – Recognizes finger counts to trigger different animations (1 finger = bubbles, 2 fingers = rain).
- Rain Animation – Simulates falling rain with white drops dynamically resetting when they reach the bottom.
- Bubble Animation – Beautiful bubbles float up the screen, complete with a shiny highlight effect.
- Live Webcam Integration – Captures and processes real time video input using OpenCV.

---

### 🌍 Tech Stack
- Python 3
- Selenium
- OpenCV – for video processing and drawing animations.
- MediaPipe – for hand landmark detection and face mesh tracking.
- NumPy – for random generation and matrix operations.
- Computer Vision – for gesture recognition and animation logic.

---

## Project Structure

```
gesture-control-app
├── src
│   ├── main.py                # Entry point of the application
│   ├── gesture_detector.py     # Contains the GestureDetector class
│   ├── animations
│   │   ├── balloon.py         # Logic for balloon animation
│   │   └── rain.py            # Logic for rainy background
│   └── utils
│       ├── mediapipe_utils.py  # MediaPipe utility functions
│       └── cv_utils.py        # Computer vision utility functions
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
└── tests
    └── test_gesture_detector.py # Unit tests for gesture detection
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd gesture-control-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python src/main.py
   ```

## Usage

- **One Finger Gesture**: Displays a moving balloon animation.
- **Two Fingers Gesture**: Displays a rainy background.

## Contributing

Feel free to submit issues or pull requests for improvements or bug fixes.
