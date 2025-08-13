
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyautogui

# Load trained model
with open("model/gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("🎥 Starting gesture detection. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            X = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(X)[0]

            cv2.putText(frame, f"Gesture: {prediction}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # Map gestures to actions
            if prediction == "left_click":
                pyautogui.click()
            elif prediction == "right_click":
                pyautogui.click(button='right')
            elif prediction == "scroll":
                pyautogui.scroll(-50)
            elif prediction == "hover":
                x, y = pyautogui.position()
                pyautogui.moveTo(x + 10, y)

    cv2.imshow("Hand Gesture Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
