import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Title
st.title("🎓 Student Score Predictor")

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# Features and target
X = df[[
    "parental level of education",
    "test preparation course",
    "reading score",
    "writing score"
]]
y = df["math score"]

# Encoding
X = pd.get_dummies(X, drop_first=True)

# Train model
model = LinearRegression()
model.fit(X, y)

# User inputs
reading = st.number_input("Enter Reading Score", 0, 100)
writing = st.number_input("Enter Writing Score", 0, 100)

parent_edu = st.selectbox(
    "Parental Level of Education",
    df["parental level of education"].unique()
)

test_prep = st.selectbox(
    "Test Preparation Course",
    df["test preparation course"].unique()
)

# Convert input to dataframe
input_data = pd.DataFrame({
    "reading score": [reading],
    "writing score": [writing],
    "parental level of education": [parent_edu],
    "test preparation course": [test_prep]
})

# Encode input
input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=X.columns, fill_value=0)

# Prediction
if st.button("Predict"):
    result = model.predict(input_data)
    st.success(f"Predicted Math Score: {round(result[0], 2)}")