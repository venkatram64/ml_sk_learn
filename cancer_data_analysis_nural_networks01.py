import mglearn
import graphviz
from sklearn.tree import export_graphviz

import matplotlib.pyplot as plot
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


mglearn.plots.plot_logistic_regression_graph()

cancer = load_breast_cancer()

X_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=0)

mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)

print("Accuracy on the training subset {:.3f}".format(mlp.score(X_train, y_train)))
print("Accuracy on the test subset {:.3f}".format(mlp.score(x_test, y_test)))
