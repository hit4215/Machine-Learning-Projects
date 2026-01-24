import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

df = pd.read_csv('gold.csv')
print(df.head())

print(df.shape)
#print(df.info())
print(df.describe())

df.drop(columns='Date',inplace=True)
#sns.heatmap(df.corr(),annot=True)
#plt.show()

#sns.distplot(df['GLD'],color='green')
#plt.show()

x = df.drop('GLD',axis=1)
y = df['GLD']

# scale = StandardScaler()
# x_scaled_data = scale.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = RandomForestRegressor()
model.fit(x_train,y_train)
model_pred = model.predict(x_test)
#print(model_pred)

error_score = metrics.r2_score(model_pred,y_test)
print("R squared error : ", error_score)

#comapre the actual value and predicted value
y_test = list(y_test)
plt.plot(y_test,color='blue',label='Actual Value')
plt.plot(model_pred,color='green',label='Prediction Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()

new_data = np.array([[ 
   1409.130005,75.25,15.52,1.466405
]])

prediction = model.predict(new_data)
print("Predicted Gold Price:", prediction[0])
