---
title: Gender Classification Problem
date: 2018-11-21
tags: [machine learning, data science, gender classification]
excerpt: "Gender Classification challenge"
---
This will be a series of blog posts with Siraj Raval coding challenges.

Starting from Python for Data Analysis.
First Challenge is Gender classification problem.

This problem describes a person as Male or Female given their body Weight and Height.

[Siraj Raval Video](https://www.youtube.com/watch?v=T5pRlIbr6gg&list=PL2-dafEMk2A6QKz1mrk1uIGfHkC1zZ6UU)

The Data in this video is self generated and very less, so I have found another Gender classification dataset on Kaggle

[Dataset](https://www.kaggle.com/hb20007/gender-classification)

This dataset categorizes gender based on:
1. Favorite color
2. Favorite Music genre
3. Favorite Beverage
4. Favorite Soft Drink

Step 1: First we'll be importing some dependencies
```python
import pandas as pd
import numpy as np

from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

```
Pandas for data manipulation, Scikit-learn for classification, NumPy for mathematical calculations.

Step 2: Reading the Dataset
```python
data = pd.read_csv("Transformed Data Set - Sheet1.csv")
```

Step 3: Preprocessing the Dataset
```python
for label in ['Favorite Color','Favorite Music Genre','Favorite Beverage','Favorite Soft Drink']:
    data[label] = LabelEncoder().fit_transform(data[label])
```
AutoEncoder simply converts the String lables to Numberic labels for processing.

Step 4: Splitting data for Training and Testing
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
```
80% Train data and 20% Test data.

Step 5: Train different models:
```python
clf_tree = tree.DecisionTreeClassifier()
clf_tree = clf_tree.fit(x_train,y_train)
clf_svm = SVC()
clf_svm = clf_svm.fit(x_train,y_train)
clf_perceptron = Perceptron()
clf_perceptron = clf_perceptron.fit(x_train,y_train)
clf_KNN = KNeighborsClassifier()
clf_KNN = clf_KNN.fit(x_train,y_train)
```
Step 6: Test against dataset
```python
pred_tree = clf_tree.predict(x_test)
acc_tree = accuracy_score(y_test, pred_tree) * 100
print('Accuracy for DecisionTree: {}'.format(acc_tree))

pred_svm = clf_svm.predict(x_test)
acc_svm = accuracy_score(y_test, pred_svm) * 100
print('Accuracy for SVM: {}'.format(acc_svm))

pred_perceptron = clf_svm.predict(x_test)
acc_perceptron = accuracy_score(y_test, pred_perceptron) * 100
print('Accuracy for Perceptron: {}'.format(acc_perceptron))

pred_KNN = clf_svm.predict(x_test)
acc_KNN = accuracy_score(y_test, pred_KNN) * 100
print('Accuracy for KNN: {}'.format(acc_KNN))
```

Step 6: Printing the best reults
```python
index = np.argmax([acc_tree,acc_svm, acc_perceptron, acc_KNN])
classifiers = {0: 'Tree', 1: 'SVM', 2: 'Perceptron', 3: 'KNN'}
print('Best gender classifier is {}'.format(classifiers[index]))
```
[Full Code](https://github.com/InosRahul/Siraj-Raval-Challenges)

Thanks for reading.
