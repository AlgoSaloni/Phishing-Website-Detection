import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/dataset.csv")


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("Result", axis=1)
y = df["Result"]


# ==========================================
# 3. SPLIT DATA INTO TRAINING AND TESTING
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. CREATE RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ==========================================
# 5. TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 6. WEBSITE PREDICTION FUNCTION
# ==========================================

def predict_website(features):
    prediction = model.predict([features])[0]

    if prediction == -1:
        return "Phishing Website"
    else:
        return "Legitimate Website"


# ==========================================
# 7. TEST MODEL
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 8. CALCULATE ACCURACY
# ==========================================

accuracy = accuracy_score(y_test, y_pred)


print("Phishing Website Detection Model")
print("---------------------------------")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Accuracy:", round(accuracy * 100, 2), "%")


# ==========================================
# 9. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


# ==========================================
# 11. CONFUSION MATRIX VISUALIZATION
# ==========================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Phishing", "Legitimate"],
    yticklabels=["Phishing", "Legitimate"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Phishing Website Detection - Confusion Matrix")

plt.tight_layout()

plt.savefig("results/confusion_matrix.png")

plt.show()