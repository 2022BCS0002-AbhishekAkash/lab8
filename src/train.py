import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import json
import os

# Load dataset
df = pd.read_csv("data/housing.csv")

# Target column
y = df["median_house_value"]
X = df.drop("median_house_value", axis=1)

# Handle categorical column
X = pd.get_dummies(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)

# Metrics
rmse = mean_squared_error(y_test, preds, squared=False)
r2 = r2_score(y_test, preds)

metrics = {
    "dataset_size": len(df),
    "rmse": rmse,
    "r2": r2
}

print(metrics)

# Save metrics
os.makedirs("metrics", exist_ok=True)

with open("metrics/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)