import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('solar.csv',header=None) #Mine Or Rock
print(df.head())

print(df.shape)
print(df.describe())
print(df[60].value_counts())

#m Mine r Rock
print(df.groupby(60).mean())

x = df.drop(columns=60,axis=1)
y = df[60]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.10,stratify=y,random_state=42)

model = LogisticRegression()
model.fit(x_train,y_train)

x_train_prd = model.predict(x_train)
print(accuracy_score(x_train_prd,y_train))

x_test_prd = model.predict(x_test)
print(accuracy_score(x_test_prd,y_test))

input_data = (0.0307,0.0523,0.0653,0.0521,0.0611,0.0577,0.0665,0.0664,0.1460,0.2792,0.3877,0.4992,0.4981,0.4972,0.5607,0.7339,0.8230,0.9173,0.9975,0.9911,0.8240,0.6498,0.5980,0.4862,0.3150,0.1543,0.0989,0.0284,0.1008,0.2636,0.2694,0.2930,0.2925,0.3998,0.3660,0.3172,0.4609,0.4374,0.1820,0.3376,0.6202,0.4448,0.1863,0.1420,0.0589,0.0576,0.0672,0.0269,0.0245,0.0190,0.0063,0.0321,0.0189,0.0137,0.0277,0.0152,0.0052,0.0121,0.0124,0.0055)
input_data_numpy = np.asarray(input_data)
input_rehshape =  input_data_numpy.reshape(1,-1)

prd = model.predict(input_rehshape)

if prd[0] == 'R':
    print('Rock')
else:
    print('Mine')