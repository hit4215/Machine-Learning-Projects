import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('rainfall.csv')
print(df.head())

print(df['rainfall'].value_counts())

df['rainfall'].replace({'yes':1,'no':0},inplace=True)
print(df.info())

# df['winddirection'].fillna(df['winddirection'].mode(),inplace=True)
# df['windspeed'].fillna(df['windspeed'].mode(),inplace=True)

df.drop(columns=['maxtemp', 'day' ,'temparature', 'mintemp'],inplace=True)

# sns.heatmap(df.corr(),annot=True)
# plt.show()

print(df.info())

x = df.drop('rainfall',axis=1)
y = df['rainfall']

scale = StandardScaler()
x = scale.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(x_train,y_train)
x_train_pred = model.predict(x_train)
print(accuracy_score(x_train_pred,y_train))

x_test_pred = model.predict(x_test)
print(accuracy_score(x_test_pred,y_test))

print(confusion_matrix(y_test,x_test_pred))
print(classification_report(y_test,x_test_pred))

input_data = (1015.9, 19.9, 95, 81, 0.0, 40.0, 13.7)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

std_data = scale.transform(input_data_reshaped)
prediction = model.predict(std_data)
print(prediction)
if (prediction[0]== 0):
    print('No Rainfall')
else:
    print('Rainfall')


