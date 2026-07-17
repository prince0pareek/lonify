import streamlit as st

st.set_page_config(page_title="CreditWise Loan Approval", layout="wide")

st.title("🏦 CreditWise Loan Approval System")

algo = st.selectbox(
    "Select Algorithm",
    ["Logistic Regression", "KNN", "Naive Bayes"]
)

st.subheader("Applicant Details")

credit_score = st.number_input("Credit Score", 300, 900, 700)
income = st.number_input("Applicant Income", 0, 1000000, 50000)
co_income = st.number_input("Coapplicant Income", 0, 1000000, 10000)
loan_amount = st.number_input("Loan Amount", 0, 10000000, 200000)

employment = st.selectbox(
    "Employment Status",
    ["Employed", "Self-Employed", "Unemployed"]
)

marital = st.selectbox(
    "Marital Status",
    ["Single", "Married"]
)

loan_purpose = st.text_input("Loan Purpose")

if st.button("Predict Loan Approval"):
    st.success(f"Prediction using {algo}")
    st.info("Model integration next step")
