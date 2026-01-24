import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('wine.csv')
print(df.head())

#print(df.isnull().sum())
print(df.shape)
#print(df.describe())

#plt.figure(figsize=(10,10))
#sns.heatmap(df.corr(),annot=True)
#plt.show()

x = df.drop('quality',axis=1)
y = df['quality'].apply(lambda y_value:1 if y_value >= 7 else 0)
#print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=3)

model = RandomForestClassifier()
model.fit(x_train,y_train)

x_train_data = model.predict(x_train)
print(accuracy_score(x_train_data,y_train))

y_pred = model.predict(x_test)
print(accuracy_score(y_pred,y_test))

input_data = (7.8,0.57,0.31,1.8,0.069,26.0,120.0,0.99625,3.29,0.53,9.3)

input_data_array = np.asarray(input_data)
input_data_reshape = input_data_array.reshape(1,-1)

prediction = model.predict(input_data_reshape)
if prediction[0] == 1:
    print('Good Quality Wine')
else:
    print('Bad Quality Wine')

