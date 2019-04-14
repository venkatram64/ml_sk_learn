import matplotlib.pyplot as plot
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


cancer = load_breast_cancer()

X_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=0)

svm = SVC()
svm.fit(X_train,y_train)

print("Accuracy on the training subset {:.3f}".format(svm.score(X_train, y_train)))
print("Accuracy on the test subset {:.3f}".format(svm.score(x_test, y_test)))



