import pandas as pd
import numpy as np

from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('car.data')


X = data[[
    'buying',
    'maint',
    'safety'
]].values
y = data[['class']]

# converting the data
## X
le = LabelEncoder()
for i in range(len(X[0])):
    X[:,i] = le.fit_transform(X[:,i])

## y
label_mapping = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}
y['class'] = y['class'].map(label_mapping)
y = np.array(y)


# create model
knn = neighbors.KNeighborsClassifier(n_neighbors=27, weights='distance')

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25)

knn.fit(X_train,y_train)
prediction = knn.predict(X_test)
accuracy = metrics.accuracy_score(y_test, prediction)
print("predictions: ", prediction)
print("accuracy: ", accuracy)

a = 1727
print("actual value: ", y[a])
print("predicted value: ", knn.predict(X)[a])