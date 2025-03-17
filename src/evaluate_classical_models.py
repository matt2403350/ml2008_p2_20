import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load extracted features and labels
X_test = np.load("models/features.npy")
y_test = np.load("models/labels.npy")

# Load trained models
svm = joblib.load("models/svm_model.pkl")
rf = joblib.load("models/random_forest_model.pkl")
knn = joblib.load("models/knn_model.pkl")
xgb = joblib.load("models/xgboost_model.pkl")

# Dictionary to store models and predictions
models = {"SVM": svm, "Random Forest": rf, "k-NN": knn, "XGBoost": xgb}
results = {}
accuracies = {}

# Evaluate each model
for name, model in models.items():
    print(f"Evaluating {name}...")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc * 100:.2f}%")
    print(f"{name} Classification Report:\n{classification_report(y_test, preds)}")
    results[name] = (acc, preds)
    accuracies[name] = acc

# Plot confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (name, (acc, preds)) in enumerate(results.items()):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx])
    axes[idx].set_title(f"{name} Confusion Matrix")
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# Plot accuracy comparison
plt.figure(figsize=(8, 5))
names = list(accuracies.keys())
values = list(accuracies.values())

sns.barplot(x=names, y=values, palette="viridis")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison of Classical Models")
plt.ylim(0, 1)
plt.show()