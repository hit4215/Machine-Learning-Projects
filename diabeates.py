import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

df = pd.read_csv('C:\Data D\Machine Learning Project\diabetes.csv')
print(df.head())
print(df.shape)
print(df.describe())
print(df['Outcome'].value_counts())

x = df.drop(columns='Outcome',axis=1)
y = df['Outcome']

scale = StandardScaler()
x_scaled_data = scale.fit_transform(x)
print(x_scaled_data)

x = x_scaled_data
y = df['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2,random_state=2)

model = svm.SVC(kernel='linear')
model.fit(x_train,y_train)

x_train_pred = model.predict(x_train)
print(accuracy_score(x_train_pred,y_train))

y_pred = model.predict(x_test)
print(accuracy_score(y_pred,y_test))

input_data = (6,148,72,35,0,33.6,0.627,50)

input_numpy = np.asarray(input_data)
input_reshape_data = input_numpy.reshape(1,-1)

std_data = scale.transform(input_reshape_data)
prediction = model.predict(std_data)
if prediction[0] == 0:
    print('The Person Was No Diabites')
else:
    print('The Person Was Diabites')

#Add Front End
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

st.title("🩺 Diabetes Prediction System")
st.write("Enter the patient details below to predict diabetes.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=1)

# Prediction button
if st.button("Predict"):
    input_data = np.array([
        pregnancies, glucose, blood_pressure,
        skin_thickness, insulin, bmi, dpf, age
    ]).reshape(1, -1)

    std_data = scale.transform(input_data)
    prediction = model.predict(std_data)

    if prediction[0] == 0:
        st.success("✅ The person is **NOT diabetic**")
    else:
        st.error("⚠️ The person **IS diabetic**")

# python -m streamlit run diabeates.py 
