import pandas as pd
import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plot

cancer = load_breast_cancer()

X_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

print("Accuracy of KNN n = 5, on the training set: {:.3f}".format(knn.score(X_train, y_train)))
print("Accuracy of KNN n = 5, on the training set: {:.3f}".format(knn.score(x_test, y_test)))