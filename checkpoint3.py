# -*- coding: utf-8 -*-
"""
Question2:
    bedrooms
    bathrooms
    sqft_living
    sqft_lot
    floors
    waterfront
    grade
    sqft_above
    sqft_basement
    
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures

df=pd.read_csv('kc_house_data.csv')
print(df.head())
"""
"""

X=df['bedrooms'].values[:,np.newaxis]
y=df["price"].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=30)
model=LinearRegression()
model.fit(X_train,y_train)
predicted=model.predict(X_test)

"""
"""
plt.scatter(X,y,color='r')
plt.title("linear regression")
plt.ylabel("price")
plt.xlabel("sqft_above")
plt.plot(X,model.predict(X),color='k')
plt.show()

print("MSE:",metrics.mean_squared_error(y_test,model.predict(X_test)))
print("R:square:",metrics.r2_score(y_test,model.predict(X_test)))



"""
"""
X=df[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','grade','sqft_above','sqft_basement']]
y=df["price"].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=30)
model=LinearRegression()
poly=PolynomialFeatures(degree=3)
X_train_fit=poly.fit_transform(X_train)
model.fit(X_train_fit,y_train)

X_test_=poly.fit_transform(X_test)

print("MSE:",metrics.mean_squared_error(y_test,model.predict(X_test_)))
print("R:square:",metrics.r2_score(y_test,model.predict(X_test_)))


