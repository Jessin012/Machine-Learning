import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
df = pd.read_csv(url)

df = df[['mpg', 'displacement']]
df.dropna(inplace=True)

X = df[['displacement']]
y = df['mpg']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

linear_mse = mean_squared_error(y_test, y_pred_linear)
linear_r2 = r2_score(y_test, y_pred_linear)

print("Linear Regression Results")
print("MSE :", linear_mse)
print("R2  :", linear_r2)

degree = 2

poly = PolynomialFeatures(degree=degree)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

y_pred_poly = poly_model.predict(X_test_poly)

poly_mse = mean_squared_error(y_test, y_pred_poly)
poly_r2 = r2_score(y_test, y_pred_poly)

print("\nPolynomial Regression Results")
print("Degree :", degree)
print("MSE    :", poly_mse)
print("R2     :", poly_r2)

X_curve = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(X, y, color='blue', label='Actual Data')
axes[0].plot(
    X_curve,
    linear_model.predict(X_curve),
    color='red',
    linewidth=2,
    label='Linear Regression'
)
axes[0].set_xlabel("Engine Displacement")
axes[0].set_ylabel("MPG")
axes[0].set_title("Linear Regression")
axes[0].legend()
axes[0].grid(True)

axes[0].text(
    1.05,
    0.85,
    f"MSE : {linear_mse:.2f}\nR² : {linear_r2:.3f}",
    transform=axes[0].transAxes,
    fontsize=11,
    bbox=dict(facecolor='white', edgecolor='black')
)

axes[1].scatter(X, y, color='yellow', label='Actual Data')
axes[1].plot(
    X_curve,
    poly_model.predict(poly.transform(X_curve)),
    color='red',
    linewidth=2,
    label='Polynomial Regression'
)
axes[1].set_xlabel("Engine Displacement")
axes[1].set_ylabel("MPG")
axes[1].set_title("Polynomial Regression")
axes[1].legend()
axes[1].grid(True)

axes[1].text(
    1.05,
    0.85,
    f"MSE : {poly_mse:.2f}\nR² : {poly_r2:.3f}",
    transform=axes[1].transAxes,
    fontsize=11,
    bbox=dict(facecolor='white', edgecolor='black')
)

plt.tight_layout()
plt.show()

results = pd.DataFrame({
    "Model": ["Linear Regression", "Polynomial Regression"],
    "MSE": [linear_mse, poly_mse],
    "R2 Score": [linear_r2, poly_r2]
})

print("\nPerformance Comparison")
print(results)