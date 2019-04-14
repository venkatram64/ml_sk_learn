import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plot

mglearn.plots.plot_tree_not_monotone()

cancer = load_breast_cancer()

X_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)

print("Accuracy on the training subset {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on the test subset {:.3f}".format(tree.score(x_test, y_test)))

