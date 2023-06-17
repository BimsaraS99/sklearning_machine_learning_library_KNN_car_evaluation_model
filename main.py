import pandas as pd
import numpy as np
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('car.data')
# print(data.head(5))

# define X and Y axis for the training purposes
X = data[['buying', 'maint', 'safety']].values
y = data[['class']]
# print(X, y)

# pre-processing the dataset for the training purposes - method 1 [converting string to integer]
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])  # very high = 3/ high = 2/ low = 1/ med = 0
# print(X)

# method 2 [label mapping method]
label_mapping = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
y['class'] = y['class'].map(label_mapping)
y = np.array(y)
# print(y)

# creating the model with KNN algorithm
knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# creating the ml model
knn.fit(X_train, y_train)

prediction = knn.predict(X_test)
accuracy = metrics.accuracy_score(y_test, prediction)

print("Predictions: ", prediction)
print("Accuracy: ", accuracy)

# predicting car conditions

print("Actual Value - ", y[1701], '|', "Predicted value - ", knn.predict(X)[1701])

