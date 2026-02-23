import os
import csv
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

DATA_DIR = "Data/alphabets"

data = []
labels = []

for label in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label)

    if os.path.isdir(label_path):
        for file in os.listdir(label_path):
            if file.endswith(".csv"):
                file_path = os.path.join(label_path, file)
                with open(file_path, "r") as f:
                    reader = csv.reader(f)
                    row = next(reader)
                    data.append([float(x) for x in row])
                    labels.append(label)

X = np.array(data)
y = np.array(labels)

print("Training samples:", len(X))
print("Feature size:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=True,
    stratify=y
)

model = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    solver='adam',
    max_iter=700,
    learning_rate_init=0.001,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy:", accuracy)

with open("model.p", "wb") as f:
    pickle.dump({"model": model}, f)

print("Model saved successfully.")