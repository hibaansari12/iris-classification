# 🌸 Task 1 Iris Flower Classification

A beginner-friendly machine learning project that classifies iris flowers into three species (setosa, versicolor, virginica) from four measurements, using **Logistic Regression** and a **Decision Tree**.

Built as part of the Machine Learning & Data Science Internship at Datheon.

## Problem
Given sepal length, sepal width, petal length, and petal width, predict the flower's species.

## Approach
1. Loaded the dataset with Pandas and checked for missing values, duplicates, and class balance
2. Explored the data with a Seaborn pairplot
3. Split the data 80/20 into training and test sets (stratified, `random_state=42`)
4. Scaled features with `StandardScaler` (fit on training data only to avoid data leakage)
5. Trained `LogisticRegression` and `DecisionTreeClassifier`
6. Evaluated both with accuracy, precision, recall, and F1-score
7. Plotted confusion matrices and decision boundaries (petal length vs. petal width)

## Results
| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 0.933 | 0.933 | 0.933 | 0.933 |
| Decision Tree | 0.933 | 0.933 | 0.933 | 0.933 |

Both models got 28 of 30 test flowers right. All mistakes were between versicolor and virginica, whose measurements overlap.

## Screenshots
### Decision Boundaries
<img width="1457" height="588" alt="Decision boundaries of Logistic Regression and Decision Tree" src="https://github.com/user-attachments/assets/c7e74110-3fc5-4eaa-a89f-b2df58b6a684" />

*Decision boundaries using petal length and petal width. Logistic Regression draws smooth, straight borders, while the Decision Tree draws box-shaped regions. Stars are test flowers.*

### Confusion Matrices
<img width="1485" height="482" alt="Confusion matrices for both models" src="https://github.com/user-attachments/assets/afa8cde9-9fd6-4ed5-b4e6-0eb71fd5a2f4" />

*Both models classified setosa perfectly and made 2 mistakes each, between versicolor and virginica.*

### Pairplot (Exploratory Data Analysis)
<img width="992" height="827" alt="Pairplot of the Iris dataset colored by species" src="https://github.com/user-attachments/assets/412ff7b3-9fcd-49f6-8509-06e0df4ef4f6" />

*Pairplot of all four features, coloured by species. Setosa is clearly separated, while versicolor and virginica overlap slightly.*
# Task 2 — Real Estate Housing Price Prediction (OLS vs Ridge vs Lasso)

Predicting median house values on the **California Housing** dataset and comparing **Ordinary Least Squares (OLS)**, **Ridge (L2)** and **Lasso (L1)** regression using **RMSE, MAE and R²**.

Built as part of my Machine Learning & Data Science internship at **Datheon**.

---

## Problem statement

Predict `median_house_value` for California census block groups. The task requires:

- imputing missing values
- engineering interaction features
- handling multicollinearity
- standardising numeric features and encoding categorical ones
- tuning Ridge and Lasso with `GridSearchCV`
- comparing OLS, Ridge and Lasso on RMSE, MAE and R²
- plotting predicted vs actual values and residual distributions

## Dataset

| Property | Value |
|---|---|
| Source | California Housing (1990 census), `housing.csv` |
| Rows × columns | 20,640 × 10 |
| Target | `median_house_value` |
| Categorical feature | `ocean_proximity` |
| Missing values | 207 in `total_bedrooms` |
| Note | Target is capped at $500,001 (965 rows, 4.7%) |

## Approach

1. **Explore** the data: structure, summary statistics, missing values, target distribution.
2. **Split first** (80/20, `random_state=42`) so all preprocessing is learned from training data only (no data leakage).
3. **Feature engineering**
   - Median imputation of `total_bedrooms`
   - Ratios: `rooms_per_household`, `bedrooms_per_room`, `population_per_household` (winsorised at the 0.5/99.5 percentiles)
   - Interactions: `income_x_rooms`, `income_x_age`, `lat_x_lon`, `income_sq` (built from mean-centered values to reduce collinearity)
4. **Multicollinearity**: correlation heatmap and Variance Inflation Factor (VIF), then Ridge/Lasso regularisation.
5. **Preprocessing pipeline**: `StandardScaler` for numeric features, One-Hot encoding for `ocean_proximity`.
6. **Modelling**: OLS, plus Ridge and Lasso tuned over 25 `alpha` values with 5-fold `GridSearchCV`.
7. **Evaluation** on the held-out test set, followed by residual and coefficient analysis.

## Results (test set)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| OLS | $67,418 | $49,044 | 0.653 |
| Ridge (L2) | $67,427 | $49,052 | 0.653 |
| Lasso (L1) | $67,420 | $49,046 | 0.653 |

Best alphas from `GridSearchCV`: Ridge ≈ 1.16, Lasso ≈ 0.68.

### Key findings

- **All three models perform almost identically.** With ~16,500 training rows and 19 features there is little variance for regularisation to remove, so the best alpha is small and the penalty barely changes the predictions.
- **Multicollinearity is real:** `total_bedrooms`, `households`, `total_rooms` and `population` have very high VIFs (up to ~72). Dropping them barely changes accuracy (R² 0.653 → 0.650), so they inflate coefficient variance more than they hurt predictions.
- **Regularisation shows feature importance.** In the Lasso path the tiny `ISLAND` category (5 rows) and the redundant counts are zeroed first, while `median_income`, `bedrooms_per_room` and `ocean_proximity_INLAND` persist.
- **Linear models have limits here.** The $500k cap and a visible non-linearity (cheap areas over-predicted, expensive areas under-predicted) show up in the residual plots.

## Visualisations

| | |
|---|---|
| ![Correlation heatmap](images/correlation_heatmap.png) | ![Metrics comparison](images/metrics_comparison.png) |
| ![Predicted vs actual](images/predicted_vs_actual.png) | ![Residual analysis](images/residual_analysis.png) |
| ![Alpha vs RMSE](images/alpha_vs_rmse.png) | ![Lasso path](images/lasso_path.png) |

## Repository structure

```
.
├── notebooks_regression_housing.ipynb   # full analysis (run top to bottom)
├── housing.csv                          # California Housing dataset
├── images/                              # plots used in this README
├── requirements.txt
└── README.md
```

## How to run
```bash
pip install -r requirements.txt
jupyter notebook Iris_Classification.ipynb
```

## Tools
Python, Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib

## Author
Hiba Ansari: [LinkedIn](https://www.linkedin.com/in/hiba-ansari24) | [GitHub](https://github.com/hibaansari12)
