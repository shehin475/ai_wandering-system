import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("dataset.csv")

X = data[['speed', 'distance_from_home', 'time_outside']]
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Load trained model
model = joblib.load("wandering_model.pkl")

# Predictions
y_pred = model.predict(X_test)

# Classification report
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Wandering"]
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Wandering Detection")
plt.show()
