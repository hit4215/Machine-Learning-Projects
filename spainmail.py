import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('mail.csv')
print(df.head())
print(df.info())

df = df.where((pd.notnull(df)), '')
print(df.head())

df.loc[df['Category'] == 'spam', 'Category',] = 1
df.loc[df['Category'] == 'ham', 'Category',] = 0
#span 1 ham 0
x = df['Message']
y = df['Category']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)


feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(x_train)
X_test_features = feature_extraction.transform(x_test)

# convert Y_train and Y_test values as integers

y_train = y_train.astype('int')
y_test = y_test.astype('int')

model = LogisticRegression()
model.fit(X_train_features, y_train)

train_acc= model.predict(X_train_features)
print("Training accuracy:", accuracy_score(y_train,train_acc))

predictions = model.predict(X_test_features)
accuracy = accuracy_score(y_test,predictions)
print("Accuracy:", accuracy)


input_mail = ["U dun say so early hor... U c already then say..."]
input_data_features = feature_extraction.transform(input_mail)

prediction = model.predict(input_data_features)
print(prediction)

if (prediction[0]==1):
  print('Ham mail')
else:
  print('Spam mail')

