import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

DATASET_PATH = r"C:\Users\HP\Downloads\SignLanguageApp\Sign Language Detection"
YAML_PATH = os.path.join(DATASET_PATH, "data.yaml")

# Load classes mapping from data.yaml
with open(YAML_PATH, 'r') as f:
    data_yaml = yaml.safe_load(f)

# The user requested that the thumbs-up sign (which was predicting as 'thankyou' or 'yes') should mean 'ok'
# The user also requested that 'hello' (the open palm sign) should map to a funny custom string
def map_class_name(name):
    if name in ['thankyou', 'yes']:
        return 'ok'
    elif name == 'hello':
        return 'etna mar marunga ki pranjal se mafi mangta firega'
    return name

classes = [map_class_name(name) for name in data_yaml['names']]

print("Classes mapping:", classes)

def extract_features_and_labels(split_name):
    features_list = []
    labels_list = []
    
    images_dir = os.path.join(DATASET_PATH, 'dataset_split', 'images', split_name)
    labels_dir = os.path.join(DATASET_PATH, 'dataset_split', 'labels', split_name)
    
    if not os.path.exists(images_dir):
        return features_list, labels_list
        
    for img_name in os.listdir(images_dir):
        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")
        
        if not os.path.exists(label_path):
            continue
            
        # Read the label (we assume single class in YOLO format, take the first line)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                continue
            class_id = int(lines[0].split()[0])
            class_name = classes[class_id]
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_data = []
                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y
                
                for i in range(len(hand_landmarks.landmark)):
                    p_x = hand_landmarks.landmark[i].x - wrist_x
                    p_y = hand_landmarks.landmark[i].y - wrist_y
                    hand_data.extend([p_x, p_y])
                
                # Normalize to make it scale-invariant
                max_val = max(abs(val) for val in hand_data)
                if max_val > 0:
                    hand_data = [val / max_val for val in hand_data]

                # Check for exactly 42 features
                if len(hand_data) == 42:
                    features_list.append(hand_data)
                    labels_list.append(class_name)
                    
    return features_list, labels_list

print("Extracting training data...")
X_train, y_train = extract_features_and_labels('train')
print(f"Extracted {len(X_train)} training samples.")

print("Extracting testing/validation data...")
X_test, y_test = extract_features_and_labels('test')
X_val, y_val = extract_features_and_labels('val')

# Combine test and val for evaluation
X_test.extend(X_val)
y_test.extend(y_val)
print(f"Extracted {len(X_test)} testing samples.")

if len(X_train) == 0:
    print("Error: No training data found or hand landmarks couldn't be extracted.")
    exit()

X_train = np.array(X_train)
y_train = np.array(y_train)

# Initialize and train Model (1000 estimators for strong performance)
print("Training the Random Forest Classifier with 1000 estimators...")
model = RandomForestClassifier(n_estimators=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
if len(X_test) > 0:
    y_predict = model.predict(X_test)
    score = accuracy_score(y_predict, y_test)
    print(f"Model tested with accuracy: {score * 100:.2f}%")
else:
    print("Warning: No testing data found, testing skipped.")
    
# Save the trained model
save_path = os.path.join(r"C:\Users\HP\Downloads\SignLanguageApp", 'model.p')
with open(save_path, 'wb') as f:
    pickle.dump({'model': model}, f)

print(f"Model saved to {save_path}")
