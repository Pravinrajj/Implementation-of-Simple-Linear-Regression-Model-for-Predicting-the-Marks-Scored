# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas numpy and matplotlib
2. Upload a file that contains the required data
3. find x,y using sklearn
4. Use line chart and disply the graph and print the mse, mae,rmse

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: PRAVINRAJJ GK
RegisterNumber:  212222240080
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('/content/csv.csv')
df.head()
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,-1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="red")
plt.plot(X_test,regressor.predict(X_test),color="blue")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

### Output:
#### df.head()
![image](https://user-images.githubusercontent.com/117917674/229984883-f9f68e88-2ec8-4bdf-8d09-9d102c5f4852.png)
#### df.tail()
![image](https://user-images.githubusercontent.com/117917674/229985275-55babb21-e530-46a2-82e2-4a07b93a1349.png)

#### X
![image](https://user-images.githubusercontent.com/117917674/229985301-848bd36f-db3e-489d-9474-b12d8f9ac248.png)

#### Y
![image](https://user-images.githubusercontent.com/117917674/229985362-0b0a2cd4-9843-4718-9322-63a4be5e3be6.png)
#### PREDICTED Y VALUES
![image](https://user-images.githubusercontent.com/117917674/229985422-ce2ff02c-6887-4945-a9cf-344394d6d5a3.png)
#### Actual Values
![image](https://user-images.githubusercontent.com/117917674/229985504-c4331171-9151-4701-954b-f8dbdc36b085.png)

#### Train graph
![image](https://user-images.githubusercontent.com/117917674/229179802-bd80aabf-beb0-4abd-9d02-26ee4c4eb4f3.png)

#### Test graph
![image](https://user-images.githubusercontent.com/117917674/229179959-71a7ec5e-7bc0-4e4c-8f24-b9578723bc91.png)

#### MEAN SQUARE ERROR, MEAN ABSOLUTE ERROR AND RMSE
![image](https://user-images.githubusercontent.com/117917674/229985751-7d989c42-7856-4598-8fcf-29f695d53572.png)

### Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
