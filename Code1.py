import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
#import matplotlib.pylab as plt


diabetes = datasets.load_diabetes()
diabetes.data.shape
diabetes.target.shape
diabetes.feature_names

X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=0)

model = LinearRegression()

model.fit(X_train, y_train)

print("Error:, " ,model.score(X_test, y_test))

print("Thetas are:", model.coef_)

print("Value of Y-intercept is:", model.intercept_)

model.predict(X_test)





