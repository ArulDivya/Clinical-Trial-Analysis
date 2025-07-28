
# clinical_trial_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Load sample dataset (simulate trial data)
df = pd.read_csv("clinical_trial_data.csv")  # Make sure this CSV is in the same directory

# Display basic info
print("Dataset Preview:")
print(df.head())

# -----------------------------
# Descriptive Statistics
# -----------------------------
print("\nDescriptive Statistics:")
print(df.describe())

# -----------------------------
# Kaplan-Meier Survival Curve
# -----------------------------
kmf = KaplanMeierFitter()
T = df["time_to_event"]  # time in days
E = df["event_occurred"]  # 1=event occurred, 0=censored

plt.figure(figsize=(8, 5))
kmf.fit(T, event_observed=E, label="Overall Survival")
kmf.plot()
plt.title("Kaplan-Meier Curve")
plt.xlabel("Time (Days)")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.savefig("km_curve.png")
plt.show()

# -----------------------------
# Machine Learning - Early Discontinuation Prediction
# -----------------------------
features = ["age", "baseline_lab1", "biomarker_level", "ecog_score"]
target = "early_discontinuation"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.2f}")

# Confusion Matrix Plot
conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.show()
