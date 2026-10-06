**MACHINE LEARNING IN DATA SCIENCE**

**LAB 01**

*End-to-End Machine Learning Project: House Price Prediction*

Tools: Python 3.x, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

| **Student Name:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ | **Student RollNumber:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ |
|----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|

# Lab Overview & Objectives

In this practical lab you will build a complete machine learning pipeline from raw data to a deployed model. Working with the California Housing dataset, you will apply every skill taught in the first part of the course: exploratory analysis, data preprocessing, supervised learning algorithms, and model evaluation. Learning objectives of this lab are:

- Execute a full end-to-end ML workflow independently

- Perform exploratory data analysis and meaningful visualization

- Engineer features and build preprocessing pipelines with Scikit-learn

- Train, compare, and fine-tune multiple regression and classification models

- Evaluate model performance using appropriate metrics and avoid overfitting

- Interpret model results and communicate findings clearly

# Grading Rubric

| **Section**                               | **Points** | **Criteria**                   |
|-------------------------------------------|------------|--------------------------------|
| Part 1: Setup Environment                 | 10 pts     | Completeness                   |
| Part 2: Exploratory Data Analysis         | 15 pts     | Depth & insight of analysis    |
| Part 3: Data Preprocessing Pipeline       | 20 pts     | Correct implementation         |
| Part 4: Regression Models                 | 25 pts     | Training, tuning, metrics      |
| Part 5: Classification Models             | 20 pts     | Algorithm comparison & metrics |
| Part 6: Report & Interpretation           | 10 pts     | Clarity & professionalism      |
| Part 7: Bonus – Getting familiar with Git | 5 pts      | Correct implementation         |
| TOTAL                                     | 100 pts    |                                |

# PART 1 — Environment Setup (10pts)

## 1.1 Environment Setup

Create a dedicated Python environment and verify all required libraries are correctly installed.

``` python
# Run this cell to verify your environment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn import __version__ as sk_version

print(f'NumPy:       {np.__version__}')
print(f'Pandas:      {pd.__version__}')
print(f'Scikit-learn:{sk_version}')
print('All libraries imported successfully!')
```

# PART 2 — Exploratory Data Analysis (15 pts)

# Why Choose the California Housing Dataset?

# Before starting the coding part, think about why this lab uses the California Housing dataset instead of another dataset. This is an intentional design decision — understanding the reason will help you learn more effectively.

# Characteristics that Make This Dataset Suitable for the Lab

- Clean dataset with no missing values: students can focus on learning algorithms instead of spending time on manual data cleaning.

- Moderate size (~20,000 samples): large enough to demonstrate meaningful differences between models, but not so large that training time becomes a major issue.

- Supports both regression and classification tasks: the continuous target variable (MedHouseVal) can also be transformed into classification labels, allowing the dataset to support both halves of the lab.

- Features with real-world meaning: variables such as income, geographic location, and house age are intuitive and can be interpreted using domain knowledge.

# Limitations — Why It Is Not Ideal for a Real-World Assignment

- No categorical features or missing values: important preprocessing tasks such as encoding and advanced imputation are absent.

- Low target complexity and noise: the prediction problem is relatively straightforward, whereas real-world datasets are often noisier and more complex.

- Data collected from the 1990s: it does not reflect the current housing market. In Assignment 2, you will be required to find a dataset that is more appropriate for a real-world problem.

## 2.1 Dataset Overview

Load the full California Housing dataset into a Pandas DataFrame and provide a structured overview. (3 pts)

``` python
df = housing.frame  # Contains both features and target

# TODO: Display shape, column types, missing values, and descriptive statistics
print('Shape:', df.shape)
print(df.dtypes)
print(df.isnull().sum())
df.describe()
```

Answer the following questions based on your output:

- How many instances and features does the dataset contain?

- Are there any missing values? If yes, which features are affected?

- What is the range of MedInc (median income) in the dataset?

## 2.2 Visualizations

Create the following four plots. For each, write two sentences interpreting the key insight. (8 pts — 2 pts each)

1.  Distribution plot (histogram + KDE) of the target variable MedHouseVal.

2.  Correlation heatmap of all numerical features.

3.  Scatter plot of MedInc vs. MedHouseVal, colored by HouseAge.

4.  Box plots of MedHouseVal for three geographic regions: coastal (longitude \< -121), central, and inland (longitude \> -118).

``` python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# TODO 1: Distribution of target

# TODO 2: Correlation heatmap

# TODO 3: Scatter MedInc vs MedHouseVal

# TODO 4: Box plots by region

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=120)
plt.show()
```

**Interpretation / Insights:**

## 2.3 Feature Correlation Analysis

Identify the three features most correlated with MedHouseVal. Justify why these features might have predictive power based on domain knowledge. (4 pts)

``` python
# TODO: Compute and sort absolute correlations with target
correlations = df.corr()['MedHouseVal'].drop('MedHouseVal').abs().sort_values(ascending=False)
print(correlations.head(5))
```

**Top 3 features & justification:**

# PART 3 — Data Preprocessing Pipeline (20 pts)

## 3.1 Train / Test Split

Split the data into training and test sets using a stratified approach based on income category. (3 pts)

``` python
from sklearn.model_selection import train_test_split
import pandas as pd

# Create income category bins for stratification
df['income_cat'] = pd.cut(df['MedInc'], bins=[0, 1.5, 3.0, 4.5, 6.0, float('inf')],
                          labels=[1, 2, 3, 4, 5])

# TODO: Perform a stratified train/test split (80/20, random_state=42)
train_set, test_set = ...

# TODO: Remove the income_cat column from both sets

print(f'Training set: {len(train_set)} samples')
print(f'Test set:     {len(test_set)} samples')
```

Why is stratified sampling preferred over a random split here? (2 pts)

## 3.2 Feature Engineering

Before looking at the hints below, think carefully on your own: Based on the available features (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude), what new features could you create by combining the existing columns? Which new features might capture information that the original features do not clearly represent?

Write your ideas in a Markdown cell in the notebook before continuing to read.

Hint: Below are three suggested features for this lab — compare them with your own ideas and explain why they may provide predictive value. (3 pts)

Create three new engineered features that may improve model performance. (2 pts)

``` python
def add_features(df):
    df = df.copy()
    # TODO 1: Rooms per household
    df['rooms_per_household'] = ...
    # TODO 2: Bedrooms per room ratio
    df['bedrooms_ratio'] = ...
    # TODO 3: Population per household
    df['population_per_household'] = ...
    return df

train_set = add_features(train_set)
test_set  = add_features(test_set)
```

For each engineered feature, briefly explain its potential predictive value. (3 pts)

**Explanation:**

- rooms_per_household:

- bedrooms_ratio:

- population_per_household:

## 3.3 Full Preprocessing Pipeline

Build a Scikit-learn Pipeline that handles numerical scaling. Then apply it to your training and test sets. (10 pts)

``` python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

feature_cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                'Population', 'AveOccup', 'Latitude', 'Longitude',
                'rooms_per_household', 'bedrooms_ratio', 'population_per_household']

X_train = train_set[feature_cols]
y_train = train_set['MedHouseVal']
X_test  = test_set[feature_cols]
y_test  = test_set['MedHouseVal']

# TODO: Build a Pipeline with SimpleImputer (median strategy) and StandardScaler
num_pipeline = Pipeline([
    ('imputer', ...),
    ('scaler',  ...),
])

# TODO: Fit on training data and transform both sets
X_train_prepared = ...
X_test_prepared  = ...

print('Pipeline applied. Training shape:', X_train_prepared.shape)
```

#  PART 4 — Regression Models (25 pts)

Train and evaluate the following three regression models. For every model, report RMSE, MAE, and R² on both training and test sets. Detect and comment on any overfitting or underfitting.

## 4.1 Linear Regression & Gradient Descent

``` python
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_model(model, X_tr, y_tr, X_te, y_te, name):
    y_tr_pred = model.predict(X_tr)
    y_te_pred = model.predict(X_te)
    print(f'\n--- {name} ---')
    print(f'Train RMSE: {np.sqrt(mean_squared_error(y_tr, y_tr_pred)):.4f}')
    print(f'Test  RMSE: {np.sqrt(mean_squared_error(y_te, y_te_pred)):.4f}')
    print(f'Train R2:   {r2_score(y_tr, y_tr_pred):.4f}')
    print(f'Test  R2:   {r2_score(y_te, y_te_pred):.4f}')

# TODO: Train a LinearRegression model
lr_model = ...
evaluate_model(lr_model, X_train_prepared, y_train, X_test_prepared, y_test, 'Linear Regression')

# TODO: Train an SGDRegressor with max_iter=1000, random_state=42
sgd_model = ...
evaluate_model(sgd_model, X_train_prepared, y_train, X_test_prepared, y_test, 'SGD Regressor')
```

Analysis: Is Linear Regression underfitting, overfitting, or well-fitted? Justify with metrics. (3 pts)

## 4.2 Polynomial Regression & Learning Curves

``` python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve

# TODO: Create a degree-2 polynomial pipeline and evaluate it
poly_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('lin_reg', LinearRegression())
])
poly_pipeline.fit(X_train_prepared, y_train)
evaluate_model(poly_pipeline, X_train_prepared, y_train, X_test_prepared, y_test, 'Polynomial Regression (d=2)')

# TODO: Plot learning curves for both Linear and Polynomial models
# Use learning_curve() with cv=5 and train_sizes=np.linspace(0.1, 1.0, 10)
```

What do the learning curves tell you about bias vs. variance for each model? (4 pts)

## 4.3 Regularized Models & Hyperparameter Tuning

``` python
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV

# TODO: Train Ridge and Lasso with default parameters
ridge = Ridge()
lasso = Lasso()

# TODO: Use GridSearchCV to find the best alpha for Ridge
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_root_mean_squared_error')
grid_search.fit(X_train_prepared, y_train)

print('Best alpha:', grid_search.best_params_)
print('Best CV RMSE:', -grid_search.best_score_)

best_ridge = grid_search.best_estimator_
evaluate_model(best_ridge, X_train_prepared, y_train, X_test_prepared, y_test, 'Best Ridge')
```

Fill in the comparison table below. (5 pts)

| **Model**         | **Train RMSE** | **Test RMSE** | **Train R²** | **Test R²** |
|-------------------|----------------|---------------|--------------|-------------|
| Linear Regression |                |               |              |             |
| SGD Regressor     |                |               |              |             |
| Polynomial (d=2)  |                |               |              |             |
| Ridge (tuned)     |                |               |              |             |
| Lasso             |                |               |              |             |

Which model performs best and why? (3 pts)

# PART 5 — Classification Models (20 pts)

Convert the regression problem into a binary classification task: predict whether a house is "expensive" (MedHouseVal ≥ 2.5) or "affordable" (MedHouseVal \< 2.5).

``` python
# Create binary labels
y_train_bin = (y_train >= 2.5).astype(int)
y_test_bin  = (y_test  >= 2.5).astype(int)

print('Class distribution (train):')
print(y_train_bin.value_counts(normalize=True))
```

Based on the output of the value_counts(normalize=True) command above, answer the following questions (write your answers in a Markdown cell in the notebook):

- What is the ratio of expensive vs. affordable houses? Is the dataset imbalanced (class imbalance)?

- How does this affect the choice of evaluation metrics? Why can accuracy be a misleading metric when class imbalance exists?

## 5.1 K-Nearest Neighbors & Naive Bayes

``` python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# TODO: Train KNN with k=5 and evaluate
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_prepared, y_train_bin)

# TODO: Train Gaussian Naive Bayes and evaluate
gnb = GaussianNB()
gnb.fit(X_train_prepared, y_train_bin)

for model, name in [(knn, 'KNN (k=5)'), (gnb, 'Naive Bayes')]:
    y_pred = model.predict(X_test_prepared)
    print(f'\n=== {name} ===')
    print(classification_report(y_test_bin, y_pred))
    print('AUC-ROC:', roc_auc_score(y_test_bin, model.predict_proba(X_test_prepared)[:, 1]))
```

Find the optimal k for KNN using cross-validation (k = 1 to 20). Plot accuracy vs. k. (4 pts)

## 5.2 Decision Trees & Random Forests

``` python
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier

# TODO: Train a Decision Tree with max_depth=5
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train_prepared, y_train_bin)

# TODO: Print the tree structure
print(export_text(dt, feature_names=feature_cols))

# TODO: Train a Random Forest with 100 trees
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_prepared, y_train_bin)

# TODO: Plot feature importance from Random Forest
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
importances.plot(kind='bar', title='Feature Importances')
```

Which features does the Random Forest consider most important? Does this align with your EDA findings in Part 2? (3 pts)

## 5.3 Support Vector Machines

``` python
from sklearn.svm import SVC

# TODO: Train SVM with RBF kernel
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
svm_rbf.fit(X_train_prepared, y_train_bin)

# TODO: Train SVM with linear kernel
svm_lin = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
svm_lin.fit(X_train_prepared, y_train_bin)

# TODO: Compare AUC-ROC for both kernels
```

Note on computational optimization for SVM: SVM with an RBF kernel has a computational complexity of O(n²) — training on the full ~16,500 training samples may take several minutes. If your computer is slow, you could randomly select a subset of 5,000 samples for demonstration purposes

Complete the classifier comparison table below. (5 pts)

| **Model**       | **Accuracy** | **Precision** | **Recall** | **AUC-ROC** |
|-----------------|--------------|---------------|------------|-------------|
| KNN (k=5)       |              |               |            |             |
| KNN (optimal k) |              |               |            |             |
| Naive Bayes     |              |               |            |             |
| Decision Tree   |              |               |            |             |
| Random Forest   |              |               |            |             |
| SVM (RBF)       |              |               |            |             |
| SVM (Linear)    |              |               |            |             |

Which classifier would you recommend for production use? Justify your choice considering both performance and computational cost. (3 pts)

# PART 6 — Report & Interpretation (10 pts)

Write a concise executive summary (200–300 words) of your findings. Address:

- The main characteristics and challenges of the California Housing dataset

- Which preprocessing steps had the greatest impact on model performance

- Your best regression model and its real-world prediction accuracy

- Your best classification model and its suitability for the task

- Two concrete recommendations for improving model performance further

# PART 7 - Bonus — Getting familiar with Git (5pts):

Execute the following three basic Git commands in your project directory before submission:

git init → initialize a repository

git add lab1_notebook.ipynb → add the notebook to the staging area

git commit -m "Lab1: complete supervised learning pipeline" → save a snapshot of your work

# Submission Checklist

**Before Submitting, Verify:**

- Jupyter Notebook (.ipynb) with all cells executed and outputs visible

- All answers and interpretations must be written directly in Markdown cells within the notebook (do not submit a separate Word document for this content). The notebook should function as a self-documenting report: anyone reading the notebook should be able to fully understand your entire workflow and analysis process.

- All plots saved as PNG files (eda_plots.png, learning_curves.png, etc.)

- Code is clean and commented

- Notebook runs end-to-end without errors (Kernel \> Restart & Run All)
