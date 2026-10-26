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
![Confusion matrices](images/confusion_matrices.png)
<img width="1485" height="482" alt="image" src="https://github.com/user-attachments/assets/afa8cde9-9fd6-4ed5-b4e6-0eb71fd5a2f4" />


## How to run
```bash
pip install -r requirements.txt
jupyter notebook Iris_Classification.ipynb
```

## Tools
Python, Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib

## Author
Hiba Ansari: [LinkedIn](https://www.linkedin.com/in/hiba-ansari24) | [GitHub](https://github.com/hibaansari12)
