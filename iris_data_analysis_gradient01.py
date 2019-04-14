import matplotlib.pyplot as plot
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#multi class classification

iris = load_iris()

print(iris.target_names)

X_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, stratify=iris.target, random_state=0)

gboost = GradientBoostingClassifier(learning_rate=0.01, random_state=0)
gboost.fit(X_train, y_train)

print("The decision function for the 3-class iris dataset:\n\n{}".format(gboost.decision_function(x_test[:10])))

print("Predicted probabilities for samples in the iris dataset \n\n{} ".format(gboost.predict_proba(x_test[:10])))

