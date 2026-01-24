import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('credit.csv')
print(df.head())
print(df.shape)

# print(df.isnull().sum())

print(df['Class'].value_counts())

#this dataset is highly unbalance 0 normal and 1 fraud

legit = df[df.Class == 0]
fraud = df[df.Class == 1]
print(legit.shape)
print(fraud.shape)

print(legit.Amount.describe())
print(fraud.Amount.describe())

#method use under sampling kem ke fraud data is so low
#Build a sample dataset containing similar distribution of normal transactions and Fraudulent Transactions
#Number of Fraudulent Transactions --> 492

legit_sample = legit.sample(n=492)

#concate two dataframe
new_datset = pd.concat([legit_sample,fraud],axis=0)
print(new_datset['Class'].value_counts())

x = new_datset.drop('Class',axis=1)
y = new_datset['Class']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LogisticRegression()
model.fit(x_train,y_train)

test_acc = model.predict(x_train)
print(accuracy_score(test_acc,y_train)*100)

y_pred = model.predict(x_test)
print(accuracy_score(y_pred,y_test)*100)

