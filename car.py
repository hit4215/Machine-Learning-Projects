import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

df = pd.read_csv('car.csv') 
print(df.head())
print(df.shape)
print(df.dtypes)
print(df['Fuel_Type'].unique())
print(df['Seller_Type'].unique())
print(df['Transmission'].unique())

df['fuel_type'] = df['Fuel_Type'].replace({'Petrol': 0, 'Diesel': 1, 'CNG': 2},inplace=True)
df['seller_type'] = df['Seller_Type'].replace({'Dealer': 0, 'Individual': 1},inplace=True)
df['transmission'] = df['Transmission'].replace ({'Manual': 0, 'Automatic': 1},inplace=True)
print(df.info())

df.drop(['Car_Name'], axis=1, inplace=True)
print(df.info())

x = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

scale = StandardScaler()
x_scaled_data = scale.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled_data, y, test_size=0.2, random_state=42)

model = Lasso()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 Score:', metrics.r2_score(y_test, y_pred))
