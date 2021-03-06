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

plot.plot(X_train.min(axis=0), 'o', label='Min')
plot.plot(X_train.max(axis=0), 'v', label='Max')
plot.xlabel("Feature Index")
plot.ylabel("Feaature Magnitude in Log Scale")
plot.yscale('log')
plot.legend(loc="upper right")
plot.show()



