from __future__ import print_function
import numpy as np
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

print("Labels:", np.unique(iris_y))
# split train and test
np.random.seed(7)
X_train, X_test, y_train, y_test = train_test_split(
iris_X, iris_y, test_size=130)
print("Training size:", X_train.shape[0], ", test size:", X_test.shape[0])

Labels: [0 1 2]
Training size: 20 , test size: 130


model = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy of 1NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))
