from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Load dataset
california = fetch_california_housing(as_frame=True)
df = california.frame

# Separate features (X) and target (y)
X = df.drop("MedHouseVal", axis=1)  # All columns except target
y = df["MedHouseVal"]  # Target column

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
## Linear Regression model
# Create the model
model = LinearRegression()

# Train (fit) the model on the training data
model.fit(X_train, y_train)

coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
print(coefficients)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print results
print(f"[Linear Model] Mean Squared Error (MSE): {mse:.2f}")
print(f"[Linear Model] Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"[Linear Model] R² Score: {r2:.2f}")

plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Actual vs Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.show()

## Random Forest Regressor model

# Step 1: Create and train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 2: Predict and evaluate
rf_preds = rf_model.predict(X_test)

# Step 3: Evaluate the new model
rf_mse = mean_squared_error(y_test, rf_preds)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_preds)

print(f"[Random Forest] RMSE: {rf_rmse:.2f}")
print(f"[Random Forest] R² Score: {rf_r2:.2f}")
