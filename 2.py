import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
housing = fetch_california_housing()

# Use only AveRooms feature
X = housing.data[:, housing.feature_names.index("AveRooms")].reshape(-1, 1)
y = housing.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Gradient Descent
# -----------------------------

# Standardize feature
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add bias term
X_train_gd = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
X_test_gd = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]

# Initialize parameters
m = len(y_train)
theta = np.zeros(2)
learning_rate = 0.01
iterations = 1000

# Gradient Descent
for i in range(iterations):
    predictions = X_train_gd.dot(theta)
    errors = predictions - y_train
    gradients = (1 / m) * X_train_gd.T.dot(errors)
    theta = theta - learning_rate * gradients

# Predictions
y_pred_gd = X_test_gd.dot(theta)

# -----------------------------
# Normal Equation
# -----------------------------

# Add bias term (without scaling)
X_train_ne = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_ne = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# Compute theta
theta_ne = np.linalg.inv(X_train_ne.T.dot(X_train_ne)).dot(X_train_ne.T).dot(y_train)

# Predictions
y_pred_ne = X_test_ne.dot(theta_ne)

# -----------------------------
# Evaluation
# -----------------------------

mse_gd = mean_squared_error(y_test, y_pred_gd)
r2_gd = r2_score(y_test, y_pred_gd)

mse_ne = mean_squared_error(y_test, y_pred_ne)
r2_ne = r2_score(y_test, y_pred_ne)

print("Gradient Descent Results")
print("MSE :", mse_gd)
print("R2 Score :", r2_gd)

print("\nNormal Equation Results")
print("MSE :", mse_ne)
print("R2 Score :", r2_ne)

# Sort values for plotting regression lines
sorted_idx = np.argsort(X_test[:, 0])

# -----------------------------
# Gradient Descent Plot
# -----------------------------

plt.figure(figsize=(8, 6))

plt.scatter(X_test, y_test, color="red", label="Actual Data")
plt.plot(
    X_test[sorted_idx],
    y_pred_gd[sorted_idx],
    color="blue",
    linewidth=2,
    label="Gradient Descent Regression Line"
)

plt.text(
    0.05,
    0.95,
    f"MSE: {mse_gd:.4f}\nR² Score: {r2_gd:.4f}",
    transform=plt.gca().transAxes,
    fontsize=11,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

plt.title("Linear Regression using Gradient Descent")
plt.xlabel("Average Rooms (AveRooms)")
plt.ylabel("Median House Value")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Normal Equation Plot
# -----------------------------

plt.figure(figsize=(8, 6))

plt.scatter(X_test, y_test, color="orange", label="Actual Data")
plt.plot(
    X_test[sorted_idx],
    y_pred_ne[sorted_idx],
    color="green",
    linestyle="--",
    linewidth=2,
    label="Normal Equation Regression Line"
)

plt.text(
    0.05,
    0.95,
    f"MSE: {mse_ne:.4f}\nR² Score: {r2_ne:.4f}",
    transform=plt.gca().transAxes,
    fontsize=11,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

plt.title("Linear Regression using Normal Equation")
plt.xlabel("Average Rooms (AveRooms)")
plt.ylabel("Median House Value")
plt.legend()
plt.grid(True)
plt.show()