import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

df = pd.read_csv('parkinsons.csv')
print(df.head())
print(df.info())

df.drop(columns=['name'], inplace=True)

# sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
# plt.show()

x = df.drop(columns=['status'],axis=1)
y = df['status']

scale = StandardScaler()
x = scale.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify=y, random_state=1)

model = svm.SVC(kernel='linear')
model.fit(x_train, y_train)

train_pred = model.predict(x_train)
print(accuracy_score(train_pred,y_train))

test_pred = model.predict(x_test)
print(accuracy_score(test_pred,y_test))

input_data = (192.81800,224.42900,168.79300,0.03107,0.00016,0.01800,0.01958,0.05401,0.11908,1.30200,0.05647,0.07940,0.13778,0.16942,0.21713,8.44100,0.625866,0.768320,-2.434031,0.450493,3.079221,0.527367)
input_data_array = np.asarray(input_data)
input_data_reshape = input_data_array.reshape(1,-1)

scaled_data = scale.transform(input_data_reshape)
prediction = model.predict(scaled_data)

if (prediction[0] == 0):
  print("The Person does not have Parkinsons Disease")
else:
  print("The Person has Parkinsons")