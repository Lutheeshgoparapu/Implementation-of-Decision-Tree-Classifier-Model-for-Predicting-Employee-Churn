# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: G.Lutheesh
RegisterNumber:  212221230029
```
```
import pandas as pd
data = pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data["salary"] = le.fit_transform(data["salary"])
data.head()

x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y = data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
### data.head():
![image](https://github.com/Lutheeshgoparapu/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94154531/9e30a82d-f41d-4744-a415-15c782f803ad)
### data.info():
![image](https://github.com/Lutheeshgoparapu/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94154531/bad3c142-c516-4686-9bb0-50a06d5540e2)
### isnull() and sum():
![image](https://github.com/Lutheeshgoparapu/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94154531/05540b2b-afdd-4ec1-88db-ee9926b8c192)
### data value counts():
![image](https://github.com/Lutheeshgoparapu/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94154531/8ec93cab-23bb-43f3-b673-865ec5976046)
### data.head() for salary:
![image](https://github.com/Lutheeshgoparapu/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94154531/d3c0614b-ae05-453c-a484-daf300910b75)
### x.head():
![image](https://github.com/Lutheeshgoparapu/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/94154531/4705c06d-b536-4a82-a231-4ed52d8c2107)
### accuracy value:
```
0.984
```
### data prediction:
```
/usr/local/lib/python3.10/dist-packages/sklearn/base.py:439: UserWarning: X does not have valid features names,warning.warn(
array([0])
```





## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
