import cv2
import mediapipe as mp
import pickle

# Load trained model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    print("Error: model.p not found. Please run train_classifier.py first.")
    exit()

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    results = hands.process(frame_rgb)
    frame_rgb.flags.writeable = True

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            hand_data = []
            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y
            
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x - wrist_x
                y = hand_landmarks.landmark[i].y - wrist_y
                hand_data.extend([x, y])
            
            # Normalize to make it scale-invariant
            max_val = max(abs(val) for val in hand_data)
            if max_val > 0:
                hand_data = [val / max_val for val in hand_data]
            
            # Predict
            try:
                # Expecting exactly 42 features
                probabilities = model.predict_proba([hand_data])[0]
                max_prob = max(probabilities)
                
                # Only predict if the model is at least 70% confident that it's a known sign
                if max_prob > 0.70:
                    predicted_class_index = list(probabilities).index(max_prob)
                    predicted_character = model.classes_[predicted_class_index]
                    
                    # Draw prediction
                    h, w, c = frame.shape
                    x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * w)
                    y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * h)
                    x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * w)
                    y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * h)
                    
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 0), 2)
                    
                    font_scale = 0.5 if len(predicted_character) > 15 else 1.0
                    thickness = 1 if len(predicted_character) > 15 else 2
                    color = (0, 0, 255) if len(predicted_character) > 15 else (0, 0, 0)
                    
                    # Make sure the text is fully visible even if x_min is too far left
                    text_x = max(10, x_min - 10)
                    cv2.putText(frame, predicted_character, (text_x, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
            except Exception as e:
                print("Prediction Error:", e)

    cv2.imshow('Real-time Sign Language Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()