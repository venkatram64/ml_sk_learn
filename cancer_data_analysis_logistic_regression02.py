import pandas as pd
import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plot

cancer = load_breast_cancer()

X_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

log_reg100 = LogisticRegression(C=100)
log_reg100.fit(X_train, y_train)

print("Accuracy on the training subset {:.3f}".format(log_reg100.score(X_train, y_train)))
print("Accuracy on the test subset {:.3f}".format(log_reg100.score(x_test, y_test)))

log_reg01 = LogisticRegression(C=0.01)
log_reg01.fit(X_train, y_train)

print("Accuracy on the training subset {:.3f}".format(log_reg01.score(X_train, y_train)))
print("Accuracy on the test subset {:.3f}".format(log_reg01.score(x_test, y_test)))

plot.plot(log_reg01.coef_.T, '0', label='C=1')
plot.plot(log_reg100.coef_.T, '^', label='C=100')
plot.plot(log_reg01.coef_.T, 'v', label='C=0.01')
plot.legend()
plot.show()


