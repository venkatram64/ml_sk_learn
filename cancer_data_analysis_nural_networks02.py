import mglearn
import graphviz
from sklearn.tree import export_graphviz

import matplotlib.pyplot as plot
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


mglearn.plots.plot_logistic_regression_graph()

cancer = load_breast_cancer()

scalar = StandardScaler()


X_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=0)

X_train_scaled = scalar.fit(X_train).transform(X_train)
x_test_scaled = scalar.fit(x_test).transform(x_test)

mlp = MLPClassifier(max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)

print("Accuracy on the training subset {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Accuracy on the test subset {:.3f}".format(mlp.score(x_test_scaled, y_test)))

mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=42)
mlp.fit(X_train_scaled, y_train)

print("Accuracy on the training subset {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Accuracy on the test subset {:.3f}".format(mlp.score(x_test_scaled, y_test)))


plot.figure(figsize=(20,5))
plot.imshow(mlp.coefs_[0], interpolation="None", cmap="GnBu")
plot.yticks(range(30), cancer.feature_names)
plot.xlabel("Columns in weight matrix")
plot.ylabel("Input feature")
plot.colorbar()
plot.show()

