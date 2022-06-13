#Importing Libraries
import numpy as np
import pandas as pd


# 50_Starups.csv contains four feature 
print('-------------------------------------------------')
print('Seprating features and dependent variable . . . ')
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Encoding categorical dataset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
print('-------------------------------------------------')
print('Encoding categorical data . . . ')
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# Spliting
from sklearn.model_selection import train_test_split
print('-------------------------------------------------')
print('splitting . . . ')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Displaying
print('-------------------------------------------------')
print('Length of X_train', len(X_train))
print(X_train)
print('-------------------------------------------------')
print('Length of X_test', len(X_test))
print(X_test)
print('-------------------------------------------------')
print('Length of y_train', len(y_train))
print(y_train)
print('-------------------------------------------------')
print('Length of y_test', len(y_test))
print(y_test)
print('-------------------------------------------------')

# Machine learns
from sklearn.linear_model import LinearRegression
print('-------------------------------------------------')
print('Machine is learning . . .')
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting
print('-------------------------------------------------')
print('Predicting . . .')
y_pred = regressor.predict(X_test)


print('-------------------------------------------------')
print('Visualising the Result')
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))
