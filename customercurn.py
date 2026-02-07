import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

df = pd.read_csv('curn.csv')
print(df.shape)
print(df.describe())
#print(df.info())
#print(df.isnull().sum())
df.drop('customerID',axis=1,inplace=True)
print(df.info())
print(df['gender'].value_counts())
print(df['Partner'].value_counts())
print(df['Dependents'].value_counts())
print(df['PhoneService'].value_counts())
print(df['MultipleLines'].value_counts())
print(df['InternetService'].value_counts())
print(df['OnlineSecurity'].value_counts())
print(df['OnlineBackup'].value_counts())
print(df['DeviceProtection'].value_counts())
print(df['TechSupport'].value_counts())
print(df['StreamingTV'].value_counts())
print(df['StreamingMovies'].value_counts())
print(df['Contract'].value_counts())
print(df['PaperlessBilling'].value_counts())    
print(df['PaymentMethod'].value_counts())
print(df['Churn'].value_counts())

encoder = LabelEncoder()
df['gender'] = encoder.fit_transform(df['gender'])
df['Partner'] = encoder.fit_transform(df['Partner'])
df['Dependents'] = encoder.fit_transform(df['Dependents'])
df['PhoneService'] = encoder.fit_transform(df['PhoneService'])
df['MultipleLines'] = encoder.fit_transform(df['MultipleLines'])
df['InternetService'] = encoder.fit_transform(df['InternetService'])
df['OnlineSecurity'] = encoder.fit_transform(df['OnlineSecurity'])
df['OnlineBackup'] = encoder.fit_transform(df['OnlineBackup'])
df['DeviceProtection'] = encoder.fit_transform(df['DeviceProtection'])
df['TechSupport'] = encoder.fit_transform(df['TechSupport'])
df['StreamingTV'] = encoder.fit_transform(df['StreamingTV'])
df['StreamingMovies'] = encoder.fit_transform(df['StreamingMovies'])
df['Contract'] = encoder.fit_transform(df['Contract'])
df['PaperlessBilling'] = encoder.fit_transform(df['PaperlessBilling'])
df['PaymentMethod'] = encoder.fit_transform(df['PaymentMethod'])
df['Churn'] = encoder.fit_transform(df['Churn'])

df['TotalCharges'] = df['TotalCharges'].replace(' ',np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
print(df.isnull().sum())
print(df.info())
df.drop('gender',axis=1,inplace=True)
# sns.heatmap(df.corr(),annot=True)
# plt.show()

x = df.drop('Churn',axis=1)
y = df['Churn']

scale = StandardScaler()
x = scale.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.svm import SVC

model2 = SVC(kernel='rbf')
model2.fit(x_train,y_train)

x_train_acc2 = model2.predict(x_train)
print(accuracy_score(y_train,x_train_acc2)*100)

y_pred2 = model2.predict(x_test)
print(accuracy_score(y_test,y_pred2)*100)


#add front end
st.set_page_config(page_title="Churn Prediction System", layout="wide")

st.title("📊 Customer Churn Prediction System")
st.write("Machine Learning model with Streamlit Frontend")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('curn.csv')
    return df

df = load_data()

st.subheader("Raw Dataset")
st.dataframe(df.head())

# Preprocessing
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

encoder = LabelEncoder()
cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    df[col] = df[col].astype(str)
    df[col] = encoder.fit_transform(df[col])

# Handle TotalCharges
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

# Drop gender as in original code
if 'gender' in df.columns:
    df.drop('gender', axis=1, inplace=True)

# Features & target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2)

# Model
model = SVC(kernel='rbf', probability=True)
model.fit(x_train, y_train)

# Accuracy
train_pred = model.predict(x_train)
test_pred = model.predict(x_test)

st.subheader("📈 Model Performance")
col1, col2 = st.columns(2)
col1.metric("Training Accuracy", f"{accuracy_score(y_train, train_pred)*100:.2f}%")
col2.metric("Testing Accuracy", f"{accuracy_score(y_test, test_pred)*100:.2f}%")

st.divider()

# Prediction UI
st.subheader("🔮 Predict Customer Churn")
st.write("Enter customer details:")

user_input = {}
for col in X.columns:
    min_val = float(np.min(X[col]))
    max_val = float(np.max(X[col]))
    mean_val = float(np.mean(X[col]))
    user_input[col] = st.number_input(col, min_val, max_val, mean_val)

input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)

if st.button("Predict Churn"):
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0].max()

    if pred == 1:
        st.error(f"⚠️ Customer Likely to Churn\nConfidence: {prob*100:.2f}%")
    else:
        st.success(f"✅ Customer Not Likely to Churn\nConfidence: {prob*100:.2f}%")

st.divider()

st.caption("Built with ❤️ using Streamlit + Scikit-Learn")

# python -m streamlit run customercurn.py

