
import cv2
import numpy as np
import os
import mediapipe as mp
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Prepare output folder
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# User input
gesture_name = input("👉 Enter gesture name (e.g., hover, scroll, left_click, right_click): ").strip().lower()
gesture_path = os.path.join(DATA_DIR, gesture_name)
os.makedirs(gesture_path, exist_ok=True)

num_samples = int(input("🔢 How many samples do you want to record? "))

# Start webcam
cap = cv2.VideoCapture(0)
sample_count = 0

print(f"\n📸 Starting capture for gesture: {gesture_name}")
print("🎯 Press 's' to save a frame")
print("❌ Press 'q' to quit early\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not captured.")
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        # Safely try processing frame
        result = hands.process(rgb)
    except Exception as e:
        print(f"⚠️ Mediapipe error: {e}")
        continue

    landmark_list = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

    # Display info
    cv2.putText(frame, f'Gesture: {gesture_name} | Samples: {sample_count}/{num_samples}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Gesture Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if landmark_list:
            file_path = os.path.join(gesture_path, f"{gesture_name}_{sample_count}.npy")
            np.save(file_path, np.array(landmark_list))
            print(f"✅ Saved sample {sample_count+1} at {file_path}")
            sample_count += 1
        else:
            print("⚠️ No hand detected – sample not saved.")

    elif key == ord('q'):
        print("👋 Exiting early.")
        break

    if sample_count >= num_samples:
        print("🎉 All samples captured.")
        break

cap.release()
cv2.destroyAllWindows()
