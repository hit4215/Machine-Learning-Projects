import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('C:\\Data D\\Machine Learning Project\\IRIS.csv')
print(df.head())
print(df.shape)
print(df.info())

print(df['species'].value_counts())
encoder = LabelEncoder()
df['species'] = encoder.fit_transform(df['species'])
print(df['species'].value_counts())

x = df.drop('species', axis=1)
y = df['species']

scaler = StandardScaler()
x_scaled_data = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled_data, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Random Forest Classifier Accuracy: {accuracy * 100:.2f}%')

model2 = LogisticRegression()
model2.fit(x_train, y_train)
y_pred2 = model2.predict(x_test)
accuracy2 = accuracy_score(y_test, y_pred2)
print(f'Logistic Regression Accuracy: {accuracy2 * 100:.2f}%')

model3 = SVC()
model3.fit(x_train, y_train)
y_pred3 = model3.predict(x_test)
accuracy3 = accuracy_score(y_test, y_pred3)
print(f'Support Vector Classifier Accuracy: {accuracy3 * 100:.2f}%')

model4 = DecisionTreeClassifier()
model4.fit(x_train, y_train)
y_pred4 = model4.predict(x_test)
accuracy4 = accuracy_score(y_test, y_pred4)
print(f'Decision Tree Classifier Accuracy: {accuracy4 * 100:.2f}%')

model5 = KNeighborsClassifier()
model5.fit(x_train, y_train)
y_pred5 = model5.predict(x_test)
accuracy5 = accuracy_score(y_test, y_pred5)
print(f'K-Nearest Neighbors Classifier Accuracy: {accuracy5 * 100:.2f}%')

input_data = (6.7,3.1,4.4,1.4)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

std_data = scaler.transform(input_data_reshaped)
prediction = model.predict(std_data)
species = encoder.inverse_transform(prediction)
print(f'The predicted species for the input data is: {species[0]}')
