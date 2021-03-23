from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale

bc = load_breast_cancer()
X = scale(bc.data)
y = bc.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20)

model = KMeans(n_clusters=3, random_state=0)
model.fit(X_train)
predictions = model.predict(X_test)
print("Predictions: ", predictions)
labels = model.labels_
print("labels: ", labels)
print("Accuracy: ", accuracy_score(y_test, predictions))
#print("Actual: ", y_test)
