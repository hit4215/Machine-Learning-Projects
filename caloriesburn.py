import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn import metrics

df = pd.read_csv('calories.csv')
df1 = pd.read_csv('calories1.csv')
print(df.head())
print(df.info())

df1 = pd.concat([df1, df['Calories']], axis=1)
print(df1.info())

df1['Gender'].replace({'male': 0, 'female': 1}, inplace=True)
print(df1.info())

df1.drop(columns=['User_ID'],inplace=True)
print(df1.info())

# sns.heatmap(df1.corr(), annot=True)
# plt.show()

x  = df1.drop('Calories', axis=1)
y = df1['Calories']

scale = StandardScaler()
x = scale.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = XGBRegressor()
model.fit(x_train, y_train)

y_train_pred = model.predict(x_train)
print('R2 Score:', metrics.r2_score(y_train, y_train_pred))

y_pred = model.predict(x_test)
print('R2 Score:', metrics.r2_score(y_test, y_pred))

input_data = (0,68,190.0,94.0,29.0,105.0,40.8)  #actual prediction = 231 
input_data_as_numpy = np.array(input_data)
input_data_reshaped = input_data_as_numpy.reshape(1,-1)
std_data = scale.transform(input_data_reshaped)

prediction = model.predict(std_data)
print('Predicted Calories:', prediction)

