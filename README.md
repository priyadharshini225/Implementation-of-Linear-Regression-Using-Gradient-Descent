# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph. 

## Program:

```

/*
Program to implement the linear regression using gradient descent.
Developed by: PRIYADHARSHINI S
RegisterNumber: 212223240129

import numpy as  np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01,num_iters=100):
  X=np.c_[np.ones(len(X1)),x1]
  theta=np.zeros(X.shape[1]).reshape(-1,1)

  for _ in range(num_iters):
    predictions=(X).dot(theta).reshape(-1,1)
    errors=(predictions-y).reshape(-1,1)        
    theta=learning=learning_rate*(1/len(X1))*X.T.dot(errors)
  return theta

data=pd.read_csv("50_Startups.csv")
print(data.head())
X=(data.iloc[1:,:-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta=linear_regression(X1_Scaled,Y1_Scaled);

new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
*/
```

## Output:
![Screenshot 2024-09-01 192851](https://github.com/user-attachments/assets/b143a0d8-310d-4f84-841d-d41c8edafcdd)

![Screenshot 2024-09-01 193357](https://github.com/user-attachments/assets/7eb74198-63c9-4c90-b102-0af96a624af2)

![Screenshot 2024-09-01 193408](https://github.com/user-attachments/assets/8489dbbc-4dbf-4fed-b196-d873e0744de6)

![Screenshot 2024-09-01 193418](https://github.com/user-attachments/assets/9e1836e6-c0bf-489f-8579-fbc3c944a95c)

![Screenshot 2024-09-01 193428](https://github.com/user-attachments/assets/37f5cbe7-58f6-4c52-8735-05ccce057bd8)

![Screenshot 2024-09-01 193438](https://github.com/user-attachments/assets/08f4b403-1dd2-45d5-933c-131c7c0f2c4e)

![Screenshot 2024-09-01 193450](https://github.com/user-attachments/assets/81334199-3acc-4411-9887-34b27096e7c9)

![Screenshot 2024-09-01 193501](https://github.com/user-attachments/assets/81a15fdf-f89b-4a92-9090-00a73d366cf0)

![Screenshot 2024-09-01 193511](https://github.com/user-attachments/assets/b045e2cf-4805-4c9e-ad97-590e1550141e)

![Screenshot 2024-09-01 193521](https://github.com/user-attachments/assets/b44086c3-7b53-4d2f-a336-97d538e5d00e)

![Screenshot 2024-09-01 193531](https://github.com/user-attachments/assets/91797413-6654-4761-8e77-7877f03a1569)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
