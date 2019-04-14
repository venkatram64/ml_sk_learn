import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import graphviz
from sklearn.tree import export_graphviz

import matplotlib.pyplot as plot

mglearn.plots.plot_tree_not_monotone()

cancer = load_breast_cancer()

X_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=0)

forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)

print("Accuracy on the training subset {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on the test subset {:.3f}".format(forest.score(x_test, y_test)))

#export_graphviz(tree, out_file='cancertree.dot', class_names=['malignant', 'benign'], feature_names=cancer.feature_names, impurity=False, filled=True)

print("Feature importances: {}".format(forest.feature_importances_))
print(type(forest.feature_importances_))
print(cancer.feature_names)

n_features = cancer.data.shape[1]
plot.barh(range(n_features), forest.feature_importances_, align='center')
plot.yticks(np.arange(n_features), cancer.feature_names)
plot.xlabel("feature importance")
plot.ylabel("feature")
plot.show()