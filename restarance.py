import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
#from sklearn.linear_model import LinearRegression

df = pd.read_csv('restarance.csv')
print(df.head())
print(df.describe())
print(df.shape)
#print(df.isnull().sum())
df.drop(columns = ['Restaurant ID','Currency','Switch to order menu'],axis=1,inplace=True)
print(df.info())

print(df['City'].value_counts())

encoder = LabelEncoder()
df['Restaurant Name'] = encoder.fit_transform(df['Restaurant Name'])
df['City'] = encoder.fit_transform(df['City'])
df['Address'] = encoder.fit_transform(df['Address'])
df['Locality'] = encoder.fit_transform(df['Locality'])
df['Locality Verbose'] = encoder.fit_transform(df['Locality Verbose'])
df['Cuisines'] = encoder.fit_transform(df['Cuisines'])
df['Has Table booking'] = encoder.fit_transform(df['Has Table booking'])
df['Has Online delivery'] = encoder.fit_transform(df['Has Online delivery'])
df['Is delivering now'] = encoder.fit_transform(df['Is delivering now'])
df['Rating color'] = encoder.fit_transform(df['Rating color'])
df['Rating text'] = encoder.fit_transform(df['Rating text'])
print(df.info())

# sns.heatmap(df.corr(),annot=True)
# plt.show()

x = df.drop('Aggregate rating',axis=1)
y = df['Aggregate rating']

scale = StandardScaler()
x = scale.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = DecisionTreeRegressor()
model.fit(x_train,y_train)

x_train_acc = model.predict(x_train)
print(r2_score(x_train_acc,y_train)*100)

y_pred = model.predict(x_test)
print(r2_score(y_pred,y_test)*100)



model2 = RandomForestRegressor()
model2.fit(x_train,y_train)

x_train_acc1 = model2.predict(x_train)
print(r2_score(x_train_acc1,y_train)*100)

y_pred1 = model2.predict(x_test)
print(r2_score(y_pred1,y_test)*100)

#new prediction
new_data = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15,16]])
new_data_scaled = scale.transform(new_data)
new_prediction = model2.predict(new_data_scaled)
print("Predicted Aggregate Rating for the new data:", new_prediction[0])
