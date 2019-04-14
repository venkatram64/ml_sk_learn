import pandas as pd
import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plot

cancer = load_breast_cancer()

X_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append((clf.score(x_test,y_test)))


plot.plot(neighbors_settings, training_accuracy, label="Accuracy o the training set")
plot.plot(neighbors_settings, test_accuracy, label="Accuracy o the training set")
plot.ylabel("Accuracy")
plot.xlabel("Number of Neighbors")
plot.legend()
plot.show()