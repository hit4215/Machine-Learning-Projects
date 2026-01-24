import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import streamlit as st 

df = pd.read_csv('loan.csv')
print(df.head())

print(df.isnull().sum())
df = df.dropna()
print(df.isnull().sum())

print(df.shape)
print(df.dtypes)

df['Loan_Status'].replace({'N':0,'Y':1},inplace=True)
print(df['Loan_Status'])

df.drop(columns = 'Loan_ID',inplace=True)
print(df.shape)
print(df.dtypes)

df['Dependents'] = df['Dependents'].replace(to_replace='3+',value=4)
df['Dependents'] = df['Dependents'].astype(int)

# convert categorical columns to numerical values
df.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)
print(df.dtypes)

x = df.drop('Loan_Status',axis=1)
y = df['Loan_Status']

scale = StandardScaler()
x_scaled_data = scale.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x_scaled_data,y,test_size=0.20,random_state=42)

model = SVC(kernel='linear')
model.fit(x_train,y_train)

x_train_acc = model.predict(x_train)
print(accuracy_score(x_train_acc,y_train))

y_pred = model.predict(x_test)
print(accuracy_score(y_pred,y_test))

input_data = (1,1,2,0,1,4006,1526,168,360,1,1)

input_numpy = np.asarray(input_data)
input_reshape_data = input_numpy.reshape(1,-1)

std_data = scale.transform(input_reshape_data)
prediction = model.predict(std_data)
if prediction[0] == 0:
    print('No')
else:
    print('Yes')


#add Frontend with Streamlit    
st.title("🏦 Loan Approval Prediction")
st.write("Enter applicant details to predict loan approval")

gender = st.selectbox("Gender", ("Male", "Female"))
married = st.selectbox("Married", ("Yes", "No"))
dependents = st.selectbox("Dependents", (0, 1, 2, 3, 4))
education = st.selectbox("Education", ("Graduate", "Not Graduate"))
self_employed = st.selectbox("Self Employed", ("Yes", "No"))

applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Amount Term", min_value=0)
credit_history = st.selectbox("Credit History", (0, 1))
property_area = st.selectbox("Property Area", ("Rural", "Semiurban", "Urban"))

# Convert inputs
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

property_area = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]

# Prediction button
if st.button("Predict Loan Status"):
    input_data = np.array([[
        gender, married, dependents, education, self_employed,
        applicant_income, coapplicant_income,
        loan_amount, loan_term,
        credit_history, property_area
    ]])

    input_scaled = scale.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")
#python -m streamlit run loan.py 