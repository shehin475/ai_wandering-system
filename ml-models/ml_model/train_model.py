import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("dataset.csv")

# Features and target
X = data[['speed', 'distance_from_home', 'time_outside']]
y = data['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------
# Decision Tree Model
# -----------------------
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

print("Decision Tree Accuracy:", dt_accuracy)

# -----------------------
# Random Forest Model
# -----------------------
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_accuracy)

# -----------------------
# Save Best Model
# -----------------------
if rf_accuracy >= dt_accuracy:
    joblib.dump(rf_model, "wandering_model.pkl")
    print("Random Forest model saved as wandering_model.pkl")
else:
    joblib.dump(dt_model, "wandering_model.pkl")
    print("Decision Tree model saved as wandering_model.pkl")
