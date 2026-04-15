import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
try:
    with open('./data.pickle', 'rb') as f:
        data_dict = pickle.load(f)
except FileNotFoundError:
    print("Error: data.pickle not found. Please run data_collection.py first.")
    exit()

data = data_dict['data']
labels = data_dict['labels']

# Ensure all samples have the same length (42 features per hand)
# Sometimes multiple hands might be captured per frame. The data_collection.py saves 42 features per hand.
# Check that all elements have length of 42
valid_indices = [i for i, d in enumerate(data) if len(d) == 42]
X = np.asarray([data[i] for i in valid_indices])
y = np.asarray([labels[i] for i in valid_indices])

if len(X) == 0:
    print("Error: No valid data elements with 42 features found.")
    exit()

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

# Initialize and train Model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f"Model trained with accuracy: {score * 100:.2f}%")

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model saved to model.p")
