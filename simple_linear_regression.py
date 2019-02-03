# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 21:56:24 2018

@author: aphees
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 1].values

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

"""from sklearn.preprocessing import StandardScalar
sc_x=StandardScalar()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)"""

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("salary vs experience")
plt.xlabel("years of experience")
plt.ylabel("salary")
plt.show()


plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("salary vs experience")
plt.xlabel("years of experience")
plt.ylabel("salary")
plt.show()