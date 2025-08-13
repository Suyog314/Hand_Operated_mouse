
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Path to the gesture data
data_dir = "data"
X = []
y = []

print("📂 Loading data from:", data_dir)

# Iterate through each gesture folder
for gesture_name in os.listdir(data_dir):
    gesture_dir = os.path.join(data_dir, gesture_name)
    if not os.path.isdir(gesture_dir):
        continue

    for file in os.listdir(gesture_dir):
        if file.endswith(".npy"):
            file_path = os.path.join(gesture_dir, file)
            landmarks = np.load(file_path)
            X.append(landmarks)
            y.append(gesture_name)

X = np.array(X)
y = np.array(y)

print(f"✅ Loaded {len(X)} samples.")

# Flatten X
X = X.reshape(len(X), -1)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
accuracy = clf.score(X_test, y_test)
print(f"✅ Model trained with accuracy: {accuracy * 100:.2f}%")

# Save the model
with open("model/gesture_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("💾 Model saved to model/gesture_model.pkl")
