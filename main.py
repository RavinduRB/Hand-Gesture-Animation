import cv2
import mediapipe as mp
import numpy as np

class BalloonAnimation:
    def __init__(self):
        self.balloon_pos = [300, 300]
        self.balloon_size = 50
        
    def update(self, frame):
        # Draw balloon
        cv2.circle(frame, 
                  (self.balloon_pos[0], self.balloon_pos[1]), 
                  self.balloon_size, 
                  (0, 0, 255), 
                  -1)
        
        # Update position (floating effect)
        self.balloon_pos[1] -= 2
        if self.balloon_pos[1] < -self.balloon_size:
            self.balloon_pos[1] = frame.shape[0] + self.balloon_size
            
        return frame

class RainAnimation:
    def __init__(self):
        self.drops = []
        self.num_drops = 100
        
    def update(self, frame):
        height, width = frame.shape[:2]
        
        # Initialize drops if empty
        if not self.drops:
            for _ in range(self.num_drops):
                x = np.random.randint(0, width)
                y = np.random.randint(0, height)
                self.drops.append([x, y])
                
        # Draw and update drops
        for drop in self.drops:
            cv2.line(frame, 
                    (drop[0], drop[1]), 
                    (drop[0], drop[1] + 10), 
                    (255, 255, 255), 
                    1)
            drop[1] += 5
            
            if drop[1] > height:
                drop[1] = 0
                drop[0] = np.random.randint(0, width)
                
        return frame

class GestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1)
        self.mp_draw = mp.solutions.drawing_utils
        
    def detect_hands(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, 
                                         self.mp_hands.HAND_CONNECTIONS)
            return results.multi_hand_landmarks[0]
        return None
        
    def count_fingers(self, landmarks):
        if not landmarks:
            return 0
            
        fingers = []
        # Thumb
        if landmarks.landmark[4].x < landmarks.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
            
        # Other fingers
        for tip in [8, 12, 16, 20]:
            if landmarks.landmark[tip].y < landmarks.landmark[tip-2].y:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return sum(fingers)

class FaceAnimation:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Load girl filter image
        self.filter_img = cv2.imread('girl_filter.png', cv2.IMREAD_UNCHANGED)
        if self.filter_img is None:
            # Create a default pink overlay if image not found
            self.filter_img = np.zeros((300, 300, 4), dtype=np.uint8)
            self.filter_img[:, :, 0] = 255  # Pink color
            self.filter_img[:, :, 3] = 128  # Semi-transparent

    def update(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get face boundaries
                h, w = frame.shape[:2]
                x_min = w
                x_max = 0
                y_min = h
                y_max = 0
                
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min = min(x_min, x)
                    x_max = max(x_max, x)
                    y_min = min(y_min, y)
                    y_max = max(y_max, y)
                
                # Add margin
                margin = 20
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(w, x_max + margin)
                y_max = min(h, y_max + margin)
                
                # Resize filter to face size
                face_width = x_max - x_min
                face_height = y_max - y_min
                if face_width > 0 and face_height > 0:
                    resized_filter = cv2.resize(self.filter_img, (face_width, face_height))
                    
                    # Create mask from alpha channel
                    alpha = resized_filter[:, :, 3] / 255.0
                    alpha = np.expand_dims(alpha, axis=-1)
                    
                    # Extract RGB channels
                    filter_rgb = resized_filter[:, :, :3]
                    
                    # Get region of interest
                    roi = frame[y_min:y_max, x_min:x_max]
                    
                    # Combine filter with original image
                    if roi.shape[:2] == filter_rgb.shape[:2]:
                        frame[y_min:y_max, x_min:x_max] = \
                            (1 - alpha) * roi + alpha * filter_rgb
        
        return frame

class BubbleAnimation:
    def __init__(self):
        self.bubbles = []
        self.num_bubbles = 15
        
    def create_bubble(self, width):
        return {
            'x': np.random.randint(0, width),
            'y': np.random.randint(400, 600),
            'size': np.random.randint(10, 40),
            'speed': np.random.randint(2, 6)
        }
        
    def update(self, frame):
        height, width = frame.shape[:2]
        
        # Initialize bubbles if empty
        if not self.bubbles:
            for _ in range(self.num_bubbles):
                self.bubbles.append(self.create_bubble(width))
        
        # Update and draw bubbles
        for bubble in self.bubbles:
            # Draw bubble
            cv2.circle(frame, 
                      (bubble['x'], int(bubble['y'])), 
                      bubble['size'], 
                      (255, 255, 255), 
                      2)
            # Add highlight effect
            cv2.circle(frame, 
                      (bubble['x'] - bubble['size']//3, 
                       int(bubble['y'] - bubble['size']//3)), 
                      bubble['size']//4, 
                      (255, 255, 255), 
                      -1)
            
            # Move bubble up
            bubble['y'] -= bubble['speed']
            
            # Reset bubble if it goes off screen
            if bubble['y'] < -bubble['size']:
                bubble.update(self.create_bubble(width))
                
        return frame

def main():
    cap = cv2.VideoCapture(0)
    detector = GestureDetector()
    bubble_animation = BubbleAnimation()
    rain_animation = RainAnimation()
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame = cv2.flip(frame, 1)
        hand_landmarks = detector.detect_hands(frame)
        finger_count = detector.count_fingers(hand_landmarks)
        
        if finger_count == 1:
            frame = bubble_animation.update(frame)
        elif finger_count == 2:
            frame = rain_animation.update(frame)
            
        cv2.imshow('Gesture Control', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()