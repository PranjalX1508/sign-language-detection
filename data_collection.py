import os
import cv2
import pickle
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

import string

# Change these to whichever classes you want to detect
classes = list(string.ascii_uppercase)
dataset_size = 100

cap = cv2.VideoCapture(0)

data = []
labels = []

for class_name in classes:
    print(f"Ready to collect data for class: {class_name}")
    print("Press 'q' when you are ready to start recording.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, f"Press 'q' to start recording for {class_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            # For simplicity, extract the first hand found or iterate over all
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                hand_data = []
                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y
                
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x - wrist_x
                    y = hand_landmarks.landmark[i].y - wrist_y
                    hand_data.extend([x, y])
                
                data.append(hand_data)
                labels.append(class_name)
                
            counter += 1

        cv2.putText(frame, f"Recording {class_name}: {counter}/{dataset_size}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data collection completed and saved to data.pickle.")
