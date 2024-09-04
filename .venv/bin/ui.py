import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load your pre-trained model
voting_model = joblib.load('./.venv/bin/voting_model.pkl')
joblib.dump(voting_model,'./.venv/bin/voting_model.pkl')

def ensemble_predict(features):
    # Convert input features into a numpy array
    features = np.array([features])
    
    # Use the ensemble model to make predictions
    prediction = voting_model.predict(features)[0]
    
    # Return the prediction (for example, 1 for "Subscribed", 0 for "Not Subscribed")
    if prediction == 0 :
        return "Not Subscribed" 
    else :
        return "Subscribed"
# Streamlit interface
st.title("Bank Product Subscription Prediction - Ensemble Model")
st.write("Enter customer data to predict whether they will subscribe to the bank product using the ensemble model.")

# Collect user inputs
age = st.slider("Age", min_value=18, max_value=100, value=30)
marital = st.selectbox("Marital Status", options=['single', 'married', 'divorced'])
education = st.selectbox("Education", options=['tertiary', 'primary', 'secondary', 'unknown'])
contact = st.selectbox("Contact Type", options=['cellular', 'telephone', 'unknown'])
loan = st.selectbox("Loan", options=['yes', 'no'])
job = st.selectbox("Job Type", options=[
    'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
    'retired', 'self-employed', 'services', 'student', 'technician', 
    'unemployed', 'unknown'
])
default = st.selectbox("Has Credit in default?",options = ['yes','no'])
balance = st.number_input("What is the Average Yearly Balance ?",min_value = -8019,max_value = 102127,value = 1000 )
housing = st.selectbox("Do they have a housing loan?", options = ['yes','no'])
day = st.number_input("Mention the last date of contact ", min_value = 1,max_value = 31,value = 27)
month = st.selectbox("Last date of contact of which month?", options = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
duration = st.number_input("Last contact duration of call(in seconds)", min_value=0,max_value = 4918,value = 60) 
campaign = st.number_input("Number of contacts performed during this campaign and for this client ",min_value=0,max_value = 63,value=10)
pdays = st.number_input("Number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)",min_value=-1,max_value = 871 ,value = -1)
previous = st.number_input("Number of contacts performed before this campaign and for this client ",min_value = 0,max_value = 275,value = 50 )
poutcome = st.selectbox("What was the customer response to previous campaign",options= ['success','failure','unknown','other'])



# Combine inputs into a DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'marital': [marital],
    'education': [education],
    'contact': [contact],
    'loan': [loan],
    'job': [job],
    'housing':[housing],
    'default':[default],
    'balance':[balance],
    'day':[day],
    'month':[month],
    'duration':[duration],
    'campaign':[campaign],
    'pdays':[pdays],
    'previous':[previous],
    'poutcome':[poutcome]

})

# Example categorical columns
categorical_cols = ['marital', 'education', 'contact', 'loan', 'job','default','housing','month','poutcome']

# Encode categorical variables
label_encoder = LabelEncoder()
for col in categorical_cols:
    input_data[col] = label_encoder.fit_transform(input_data[col])

# Make the prediction
if st.button("Predict"):
    features = input_data.iloc[0].tolist()  # Convert the DataFrame row to a list
    result = ensemble_predict(features)
    st.write(f"Prediction Result: **{result}**")



