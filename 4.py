import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear_model = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

linear_mse = mean_squared_error(y_test, y_pred_linear)
linear_r2 = r2_score(y_test, y_pred_linear)

ridge_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge())
])

ridge_params = {
    'model__alpha': [0.01, 0.1, 1, 10, 100]
}

ridge_cv = GridSearchCV(
    ridge_pipeline,
    ridge_params,
    cv=5,
    scoring='neg_mean_squared_error'
)

ridge_cv.fit(X_train, y_train)
y_pred_ridge = ridge_cv.predict(X_test)

ridge_mse = mean_squared_error(y_test, y_pred_ridge)
ridge_r2 = r2_score(y_test, y_pred_ridge)

lasso_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Lasso(max_iter=10000))
])

lasso_params = {
    'model__alpha': [0.001, 0.01, 0.1, 1, 10]
}

lasso_cv = GridSearchCV(
    lasso_pipeline,
    lasso_params,
    cv=5,
    scoring='neg_mean_squared_error'
)

lasso_cv.fit(X_train, y_train)
y_pred_lasso = lasso_cv.predict(X_test)

lasso_mse = mean_squared_error(y_test, y_pred_lasso)
lasso_r2 = r2_score(y_test, y_pred_lasso)

results = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge Regression', 'Lasso Regression'],
    'Best Alpha': [
        'N/A',
        ridge_cv.best_params_['model__alpha'],
        lasso_cv.best_params_['model__alpha']
    ],
    'MSE': [linear_mse, ridge_mse, lasso_mse],
    'R2 Score': [linear_r2, ridge_r2, lasso_r2]
})

print("\nModel Comparison:")
print(results)

print("\nBest Ridge Alpha:", ridge_cv.best_params_['model__alpha'])
print("Best Lasso Alpha:", lasso_cv.best_params_['model__alpha'])

fig, ax = plt.subplots(1, 3, figsize=(18, 5))

bars1 = ax[0].bar(results["Model"], results["MSE"], color=["blue", "green", "orange"])
ax[0].set_title("Mean Squared Error")
ax[0].set_ylabel("MSE")
for bar in bars1:
    ax[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{bar.get_height():.2f}",
        ha="center",
        va="bottom"
    )

bars2 = ax[1].bar(results["Model"], results["R2 Score"], color=["blue", "green", "orange"])
ax[1].set_title("R² Score")
ax[1].set_ylabel("R² Score")
for bar in bars2:
    ax[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{bar.get_height():.3f}",
        ha="center",
        va="bottom"
    )

ax[2].axis("off")

table = ax[2].table(
    cellText=results.values,
    colLabels=results.columns,
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.4, 2)

ax[2].set_title("Model Comparison Table")

plt.tight_layout()
plt.show()