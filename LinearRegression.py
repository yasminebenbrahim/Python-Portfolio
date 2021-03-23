from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
#from matplotlib import pyplot as plt

boston = datasets.load_boston()

X = boston.data
y = boston.target

print("X: ", X)
print("y: ", y)

l_reg = linear_model.LinearRegression()

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3)

model = l_reg.fit(X_train,y_train)
predictions = model.predict(X_test)
print("Predictions: ", predictions)
print("R^2 value: ", l_reg.score(X,y))
