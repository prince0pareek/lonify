import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load models
log_model = pickle.load(open("log_model.pkl", "rb"))
knn_model = pickle.load(open("knn_model.pkl", "rb"))
nb_model = pickle.load(open("nb_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="CreditWise Loan Approval", layout="wide")

st.title("🏦 CreditWise Loan Approval System")

algo = st.selectbox(
    "Select Algorithm",
    ["Logistic Regression", "KNN", "Naive Bayes"]
)

# Inputs
Applicant_Income = st.number_input("Applicant Income", min_value=0)
Coapplicant_Income = st.number_input("Coapplicant Income", min_value=0)
Age = st.number_input("Age", min_value=18, max_value=100)
Dependents = st.number_input("Dependents", min_value=0)
Existing_Loans = st.number_input("Existing Loans", min_value=0)
Savings = st.number_input("Savings", min_value=0)
Collateral_Value = st.number_input("Collateral Value", min_value=0)
Loan_Amount = st.number_input("Loan Amount", min_value=0)
Loan_Term = st.number_input("Loan Term (Months)", min_value=1)

Education_Level = st.selectbox("Education Level", [0, 1])

Employment = st.selectbox(
    "Employment Status",
    ["Salaried", "Self-employed", "Unemployed"]
)

Marital = st.selectbox(
    "Marital Status",
    ["Married", "Single"]
)

Loan_Purpose = st.selectbox(
    "Loan Purpose",
    ["Car", "Education", "Home", "Personal"]
)

Property_Area = st.selectbox(
    "Property Area",
    ["Rural", "Semiurban", "Urban"]
)

Gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

Employer_Category = st.selectbox(
    "Employer Category",
    ["Government", "MNC", "Private", "Unemployed"]
)

Credit_Score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=700
)

if st.button("Predict Loan Approval"):

    # Feature Engineering
    DTI_Ratio = Loan_Amount / max(Applicant_Income, 1)
    DTI_Ratio_sq = DTI_Ratio ** 2
    Credit_Score_sq = Credit_Score ** 2

    row = {
        "Applicant_Income": Applicant_Income,
        "Coapplicant_Income": Coapplicant_Income,
        "Age": Age,
        "Dependents": Dependents,
        "Existing_Loans": Existing_Loans,
        "Savings": Savings,
        "Collateral_Value": Collateral_Value,
        "Loan_Amount": Loan_Amount,
        "Loan_Term": Loan_Term,
        "Education_Level": Education_Level,

        "Employment_Status_Salaried": int(Employment=="Salaried"),
        "Employment_Status_Self-employed": int(Employment=="Self-employed"),
        "Employment_Status_Unemployed": int(Employment=="Unemployed"),

        "Marital_Status_Single": int(Marital=="Single"),

        "Loan_Purpose_Car": int(Loan_Purpose=="Car"),
        "Loan_Purpose_Education": int(Loan_Purpose=="Education"),
        "Loan_Purpose_Home": int(Loan_Purpose=="Home"),
        "Loan_Purpose_Personal": int(Loan_Purpose=="Personal"),

        "Property_Area_Semiurban": int(Property_Area=="Semiurban"),
        "Property_Area_Urban": int(Property_Area=="Urban"),

        "Gender_Male": int(Gender=="Male"),

        "Employer_Category_Government": int(Employer_Category=="Government"),
        "Employer_Category_MNC": int(Employer_Category=="MNC"),
        "Employer_Category_Private": int(Employer_Category=="Private"),
        "Employer_Category_Unemployed": int(Employer_Category=="Unemployed"),

        "DTI_Ratio_sq": DTI_Ratio_sq,
        "Credit_Score_sq": Credit_Score_sq
    }

    df = pd.DataFrame([row])

    scaled_data = scaler.transform(df)

    if algo == "Logistic Regression":
        pred = log_model.predict(scaled_data)[0]

    elif algo == "KNN":
        pred = knn_model.predict(scaled_data)[0]

    else:
        pred = nb_model.predict(scaled_data)[0]

    if pred == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
