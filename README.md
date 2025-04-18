# README.md

# Gesture Control App

This project is a gesture-controlled application that utilizes Python, MediaPipe, OpenCV, and Selenium to create interactive animations based on hand gestures. The application recognizes gestures using computer vision techniques and displays different animations based on the number of fingers detected.

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