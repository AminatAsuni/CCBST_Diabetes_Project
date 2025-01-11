import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes

st.title("Diabetes App")
st.write("Hello World , please use app to predict the likelihood of diabetes.")

#load the data set
diab = load_diabetes()
df = pd.DataFrame(diab.data, columns=diab.feature_names)
df['Outcome'] = (diab.target > diab.target.mean()).astype(int) 

X=df.drop('Outcome',axis=1) #Input Features
y=df['Outcome'] #target Features

st.subheader("Data Overview of the first 10 rows")
st.dataframe(df.head(10))

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size =0.2,random_state=42)

#scaling tha data 
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

lr = LogisticRegression() #initiate the model
lr.fit(X_train_sc,y_train) #train the model

#make predictions
y_pred =lr.predict(X_test_sc) 

#evaluate the model
ac= accuracy_score(y_test,y_pred)

st.subheader("Model Performance Metrics")
st.write(f"Accuracy Score: {ac:.2f}")

# Display confusion matrix and classification report
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))

st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Interactive section: Predict on user input
st.subheader("Predict Diabetes Likelihood for New Data")
user_data = []
for feature in X.columns:
    value = st.number_input(f"Enter {feature}:", value=0.0)
    user_data.append(value)

if st.button("Predict"):
    user_data = scaler.transform([user_data])
    prediction = lr.predict(user_data)
    probability = lr.predict_proba(user_data)
    st.write("Prediction:", "Diabetic" if prediction[0] == 1 else "Non-Diabetic")
    st.write(f"Probability of being diabetic: {probability[0][1]:.2f}")
