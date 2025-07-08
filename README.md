# Predictive-Model

## 🧠 Machine Learning Projects

### 1. **Breast Cancer Detection**

* **Dataset**: [UCI Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data)
* **Description**: [Feature Info](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names)

**Overview**:
This project focuses on predicting breast cancer diagnoses (malignant vs. benign) using the Wisconsin Diagnostic dataset. The features include statistical properties such as mean, standard error, and worst-case measurements for various cell nuclei characteristics. Models implemented include Decision Tree, K-Nearest Neighbors (KNN), Logistic Regression, and Support Vector Machines (SVM).

---

### 2. **Car Evaluation Classification**

* **Dataset**: [UCI Car Evaluation Dataset](http://archive.ics.uci.edu/ml/datasets/Car+Evaluation)

**Overview**:
This project compares classification algorithms (Decision Tree, KNN, Logistic Regression, SVM, and Naive Bayes) on a dataset containing 1,728 car records. Each record includes attributes such as buying price, maintenance cost, number of doors, seating capacity, luggage size, and safety level. The final output is a classification of the car’s acceptability. Categorical variables were encoded and transformed as part of preprocessing.

---

### 3. **Consumer Spending Prediction with Multi-Model Regression**

**Problem Statement**:
Developed and benchmarked several regression models—including Linear Regression, k-NN, Regression Trees, SVM, Neural Networks, and Ensemble techniques—to predict consumer spending behavior in response to catalog mailings. The analysis covered both the full dataset and a filtered subset of actual purchasers. Hyperparameter tuning and normalization were applied to assess the effect of purchase filtering on predictive performance.

---

### 4. **Cost-Sensitive Spam Email Detection**

* **Dataset**: [UCI Spambase Dataset](https://archive.ics.uci.edu/dataset/94/spambase)

**Problem Statement**:
Built and optimized machine learning classifiers to detect spam emails. Two approaches were evaluated: one optimized for overall accuracy, and another cost-sensitive model minimizing false negatives (10:1 penalty over false positives). Techniques used included k-NN, Decision Trees, Naive Bayes, SVM, and Ensemble methods. Evaluation was performed using nested cross-validation, confusion matrices, ROC curves, lift charts, and cost analysis.

---

### 5. **Function Approximation with Shallow vs. Deep Neural Networks**

**Problem Statement**:
This study explores how neural networks of varying depth approximate the nonlinear function
\$f(x) = 2(2\cos^2(x) - 1)^2 - 1\$. Using 120,000 data points sampled uniformly from $\[-2\pi, 2\pi]\$, networks with 1, 2, and 3 hidden layers were trained. Performance was evaluated using mean squared error. Results showed that deeper networks achieved lower error with fewer parameters, illustrating their superior capacity for learning complex patterns.

![Function Approximation Result](https://github.com/user-attachments/assets/3db37477-84e6-4a6a-9da6-2516a3d64cf5)

---


