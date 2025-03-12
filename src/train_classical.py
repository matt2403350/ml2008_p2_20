import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import joblib

# Load extracted features and labels
features = np.load("models/features.npy")
labels = np.load("models/labels.npy")

# Standardize features (important for SVM & k-NN)
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train SVM
print("Training SVM...")
svm = SVC(kernel="linear")
svm.fit(X_train, y_train)

# Train Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Train k-Nearest Neighbors
print("Training k-NN...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Train XGBoost with hyperparameter tuning
print("Training XGBoost with hyperparameter tuning...")
xgb_params = {
    'n_estimators': [100, 150],
    'max_depth': [3, 5],
    'learning_rate': [0.1],
    'subsample': [0.8],
    'tree_method': ['hist']
}

grid_search = HalvingGridSearchCV(XGBClassifier(eval_metric='mlogloss', tree_method="hist"), xgb_params, cv=2, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_xgb = grid_search.best_estimator_
print("Best XGBoost Parameters:", grid_search.best_params_)

"""
print("Training XGBoost...")
xgb = XGBClassifier(eval_metric='mlogloss')
xgb.fit(X_train, y_train)
"""

# Save trained models
joblib.dump(svm, "models/svm_model.pkl")
joblib.dump(rf, "models/random_forest_model.pkl")
joblib.dump(knn, "models/knn_model.pkl")
joblib.dump(best_xgb, "models/xgboost_model.pkl")

print("All models saved successfully!")

# Evaluate SVM
svm_preds = svm.predict(X_test)
svm_acc = accuracy_score(y_test, svm_preds)
print(f"SVM Accuracy: {svm_acc * 100:.2f}%")
print("SVM Classification Report:")
print(classification_report(y_test, svm_preds))

# Evaluate Random Forest
rf_preds = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
print(f"Random Forest Accuracy: {rf_acc * 100:.2f}%")
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_preds))

# Evaluate k-NN
knn_preds = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_preds)
print(f"k-NN Accuracy: {knn_acc * 100:.2f}%")
print("k-NN Classification Report:")
print(classification_report(y_test, knn_preds))

# Evaluate XGBoost
xgb_preds = best_xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)
print(f"XGBoost Accuracy: {xgb_acc * 100:.2f}%")
print("XGBoost Classification Report:")
print(classification_report(y_test, xgb_preds))


