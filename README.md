# 🌸 Iris Flower Classification

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


## How to run
```bash
pip install -r requirements.txt
jupyter notebook Iris_Classification.ipynb
```

## Tools
Python, Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib

## Author
Hiba Ansari: [LinkedIn](https://www.linkedin.com/in/hiba-ansari24) | [GitHub](https://github.com/hibaansari12)
