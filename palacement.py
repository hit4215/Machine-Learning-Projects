import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('C:\\Data D\\Machine Learning Project\\palacement.csv')
print(df.head())
print(df.info())
print(df.shape)

df.drop(columns=['Student_ID'],axis=1,inplace=True)
print(df.info())


x = df.drop('Placement',axis=1)
y = df['Placement']

scale = StandardScaler()
x_scaled_data = scale.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled_data, y, test_size=0.2, random_state=42)
model = SVC()
model.fit(x_train,y_train)
x_train_acc = model.predict(x_train)
y_train_acc = accuracy_score(x_train_acc,y_train)   
print("Training Accuracy SVM:",y_train_acc * 100)

y_pred = model.predict(x_test)
print("SVM:",accuracy_score(y_pred,y_test) * 100)

model2 = LogisticRegression()
model2.fit(x_train,y_train)
x_train_acc = model2.predict(x_train)
y_train_acc = accuracy_score(x_train_acc,y_train)   
print("Training Accuracy Logistic Regression:",y_train_acc * 100)

y_pred2 = model2.predict(x_test)
print("Logistic Regression:",accuracy_score(y_pred2,y_test) * 100)

model3 = DecisionTreeClassifier()
model3.fit(x_train,y_train)
x_train_acc = model3.predict(x_train)
y_train_acc = accuracy_score(x_train_acc,y_train)   
print("Training Accuracy Decision Tree:",y_train_acc * 100)

y_pred3 = model3.predict(x_test)
print("Decision Tree:",accuracy_score(y_pred3,y_test) * 100)

model4 = KNeighborsClassifier()
model4.fit(x_train,y_train)
x_train_acc = model4.predict(x_train)
y_train_acc = accuracy_score(x_train_acc,y_train)
print("Training Accuracy KNN:",y_train_acc * 100)

y_pred4 = model4.predict(x_test)
print("KNN:",accuracy_score(y_pred4,y_test) * 100)

model5 = GaussianNB()
model5.fit(x_train,y_train)
x_train_acc = model5.predict(x_train)
y_train_acc = accuracy_score(x_train_acc,y_train)
print("Training Accuracy Naive Bayes:",y_train_acc * 100)

y_pred5 = model5.predict(x_test)
print("Naive Bayes:",accuracy_score(y_pred5,y_test) * 100)

x = ['SVM','Logistic Regression','Decision Tree','KNN','Naive Bayes']
y = [accuracy_score(y_pred,y_test)*100,accuracy_score(y_pred2,y_test)*100,accuracy_score(y_pred3,y_test)*100,accuracy_score(y_pred4,y_test)*100,accuracy_score(y_pred5,y_test)*100]
sns.barplot(x=x,y=y,palette='viridis')
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.title('Algorithm Comparison')
plt.show()

#actual prediction
input_data = (5.8,142)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

std_data = scale.transform(input_data_reshaped)
prediction = model4.predict(std_data)
if prediction[0] == 1:
    print("The student is placed")
else:
    print("The student is not placed")

#add front end using streamlit 
import streamlit as st
st.title("Student Placement Prediction")
st.write("Enter the details of the student to predict placement status.")
cgpa = st.number_input("Enter CGPA", min_value=0.0, max_value=10.0, step=0.1)
test_score = st.number_input("Enter Test Score", min_value=0, max_value=200, step=1)
if st.button("Predict Placement"):
    input_data = (cgpa, test_score)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    std_data = scale.transform(input_data_reshaped)
    prediction = model4.predict(std_data)
if prediction[0] == 1:
    st.success("The student is placed")
else:
    st.error("The student is not placed")