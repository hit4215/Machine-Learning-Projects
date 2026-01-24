import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('titanic.csv')
print(df.head())
df.drop(columns = ['PassengerId','Name','Ticket','Cabin'],inplace=True)
print(df.info())

print(df['Sex'].value_counts())
df.replace({'Sex':{'male':0,'female':1}},inplace=True)
print(df.info())

df['Age'].fillna(df['Age'].mean(),inplace=True)
print(df.isnull().sum())

df['Embarked'].fillna(df['Embarked'].mode(),inplace=True)
print(df.info())

print(df['Embarked'].value_counts())
encoder = LabelEncoder()
df['Embarked'] = encoder.fit_transform(df['Embarked'])
print(df.info())

x = df.drop('Survived',axis=1)
y = df['Survived']

scale = StandardScaler()
x = scale.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify=y, random_state=42)

model = LogisticRegression()
model.fit(x_train,y_train)

train_pred = model.predict(x_train)
print(accuracy_score(train_pred,y_train)*100)

test_pred = model.predict(x_test)
print(accuracy_score(test_pred,y_test)*100)

# New passenger data
new_data = np.array([[1, 1, 28, 0, 0, 80, 0]])  # Embarked='S' → 2

new_passenger_scaled = scale.transform(new_data)
prediction = model.predict(new_passenger_scaled)

if prediction[0] == 1:
    print("The passenger is likely to survive.")
else:
    print("The passenger is not likely to survive.")
