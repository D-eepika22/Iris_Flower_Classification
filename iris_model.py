from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pickle

iris=load_iris()
x=iris.data
y=iris.target
x_train,x_test ,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=RandomForestClassifier()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print("Accuracy:",accuracy_score(y_test,y_pred))

pickle.dump(model,open("iris_model.pkl","wb"))

print("\nenter flower measurements to predict the class (sepal length, sepal width, petal length, petal width):")
s1=float(input("Sepal Length: "))
s2=float(input("Sepal Width: "))
s3=float(input("Petal Length: "))
s4=float(input("Petal Width: "))

data=np.array([[s1,s2,s3,s4]])

prediction=model.predict(data)
print("predicted species:",iris.target_names[prediction[0]])