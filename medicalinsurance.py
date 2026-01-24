import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv('insurance.csv')
print(df.head())

print(df.info())
print(df.isnull().sum())
print(df.describe())

print(df['sex'].value_counts())
print(df['smoker'].value_counts())
print(df['region'].value_counts())
df.replace({'sex':{'male' : 0,'female' : 1}},inplace=True)
df.replace({'smoker':{'yes' : 0 ,'no' : 1}},inplace=True)
# encoder = LabelEncoder()
# df['region'] = encoder.fit_transform(df['region'])
df.replace({'region':{'southeast' : 0,'southwest' : 1,'northwest' : 2,'northeast' : 3}},inplace=True)
#df = pd.get_dummies(df, columns=["region"])

print(df.info())

sns.heatmap(df.corr(),annot=True)
plt.show()

x = df.drop(columns = 'charges',axis=1)
y = df['charges']

scale = StandardScaler()
x_scaled_data = scale.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x_scaled_data,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(x_train,y_train)

x_train_acc = model.predict(x_train)
r2_test = metrics.r2_score(y_train,x_train_acc)
print('R squared vale : ', r2_test)

y_pred = model.predict(x_test)
r2_test = metrics.r2_score(y_test,y_pred)
print('R squared vale : ', r2_test)


input_data = (46,1,33.44,1,1,0)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

scaled_data = scale.transform(input_data_reshaped)
prediction = model.predict(scaled_data)
print(prediction)

print('The insurance cost is USD ', prediction[0])

