# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:35:31 2024

@author: hazal
"""

####  Polynomial regression

import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Salary_Data.csv')

x = data[['YearsExperience']].values
y = data[['Salary']].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

plt.scatter(x,y,color='red')
plt.plot(x, lin_reg.predict(x),color='blue')
plt.show()


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)
plt.scatter(x, y,color = 'red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color = 'blue')
plt.show()

##### predicts

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[4.1]]))

print(lin_reg2.predict(poly_reg.fit_transform([[11]])))
print(lin_reg2.predict(poly_reg.fit_transform([[4.1]])))

