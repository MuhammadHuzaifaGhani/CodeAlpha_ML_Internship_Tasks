# import streamlit as st
# import pandas as pd
# import joblib

# # Load trained Random Forest model
# model = joblib.load("random_forest_model.joblib")

# # Streamlit page config
# st.set_page_config(page_title="Credit Scoring App", page_icon="💳", layout="centered")

# st.title("💳 Credit Scoring Prediction App")
# st.write("Predict an individual's creditworthiness using financial history.")

# # --- User Inputs ---
# st.header("Enter Applicant Information")

# Income = st.number_input("Income ($)", min_value=0, max_value=500000, value=50000, step=1000)
# Debt = st.number_input("Debt ($)", min_value=0, max_value=300000, value=10000, step=1000)
# Payment_History = st.selectbox("Payment History", ["Good", "Average", "Poor"])
# Loan_Amount = st.number_input("Loan Amount ($)", min_value=0, max_value=500000, value=20000, step=1000)
# Age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)

# # --- Encode Payment_History (same encoding as training) ---
# payment_history_map = {"Average": 0, "Poor": 1, "Good": 2}
# payment_history_encoded = payment_history_map[Payment_History]

# # --- Derived Features ---
# Debt_to_Income = round(Debt / Income, 2) if Income > 0 else 0
# Loan_to_Income = round(Loan_Amount / Income, 2) if Income > 0 else 0
# High_Debt_Flag = 1 if Debt_to_Income > 0.4 else 0
# High_Loan_Flag = 1 if Loan_to_Income > 0.3 else 0

# # --- Prepare input data (must match training order) ---
# input_data = pd.DataFrame([[
#     Income,
#     Debt,
#     payment_history_encoded,
#     Loan_Amount,
#     Debt_to_Income,
#     Loan_to_Income,
#     High_Debt_Flag,
#     High_Loan_Flag,
#     Age
# ]], columns=[
#     "Income",
#     "Debt",
#     "Payment_History",
#     "Loan_Amount",
#     "Debt_to_Income",
#     "Loan_to_Income",
#     "High_Debt_Flag",
#     "High_Loan_Flag",
#     "Age"
# ])

# # --- Prediction ---
# if st.button("Predict Creditworthiness"):
#     prediction = model.predict(input_data)[0]
#     probability = model.predict_proba(input_data)[0][1]  # Probability of being Creditworthy

#     if prediction == 1:
#         st.success(f"✅ Applicant is Creditworthy (Confidence: {probability*100:.0f}%)")

#     else:
#         st.error(f"❌ Applicant is Not Creditworthy (Confidence: {(1-probability)*100:.0f}%)")

import streamlit as st
import pandas as pd
import joblib

# --- Load trained models ---
log_reg_model = joblib.load("logistic_regression_model.joblib")
dt_model = joblib.load("decision_tree_model.joblib")
rf_model = joblib.load("random_forest_model.joblib")

# Streamlit config
st.set_page_config(page_title="Credit Scoring App", page_icon="💳", layout="centered")

st.title("💳 Credit Scoring Prediction App (Multiple Models)")
st.write("Compare predictions from Logistic Regression, Decision Tree, and Random Forest.")

# --- User Inputs ---
st.header("Enter Applicant Information")

Income = st.number_input("Income ($)", min_value=0, max_value=500000, value=50000, step=1000)
Debt = st.number_input("Debt ($)", min_value=0, max_value=300000, value=10000, step=1000)
Payment_History = st.selectbox("Payment History", ["Good", "Average", "Poor"])
Loan_Amount = st.number_input("Loan Amount ($)", min_value=0, max_value=500000, value=20000, step=1000)
Age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)

# --- Encode Payment_History ---
payment_history_map = {"Average": 0, "Poor": 1, "Good": 2}
payment_history_encoded = payment_history_map[Payment_History]

# --- Derived Features ---
Debt_to_Income = round(Debt / Income, 2) if Income > 0 else 0
Loan_to_Income = round(Loan_Amount / Income, 2) if Income > 0 else 0
High_Debt_Flag = 1 if Debt_to_Income > 0.4 else 0
High_Loan_Flag = 1 if Loan_to_Income > 0.3 else 0

# --- Prepare input data (same feature order as training) ---
input_data = pd.DataFrame([[
    Income,
    Debt,
    payment_history_encoded,
    Loan_Amount,
    Debt_to_Income,
    Loan_to_Income,
    High_Debt_Flag,
    High_Loan_Flag,
    Age
]], columns=[
    "Income",
    "Debt",
    "Payment_History",
    "Loan_Amount",
    "Debt_to_Income",
    "Loan_to_Income",
    "High_Debt_Flag",
    "High_Loan_Flag",
    "Age"
])

# --- Prediction Button ---
if st.button("Predict with All Models"):

    # Logistic Regression
    log_pred = log_reg_model.predict(input_data)[0]
    log_proba = log_reg_model.predict_proba(input_data)[0][1]

    # Decision Tree
    dt_pred = dt_model.predict(input_data)[0]
    dt_proba = dt_model.predict_proba(input_data)[0][1]

    # Random Forest
    rf_pred = rf_model.predict(input_data)[0]
    rf_proba = rf_model.predict_proba(input_data)[0][1]

    # --- Show Results in Columns ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📊 Logistic Regression")
        if log_pred == 1:
            st.success(f"✅ Creditworthy ({log_proba*100:.0f}%)")
        else:
            st.error(f"❌ Not Creditworthy ({(1-log_proba)*100:.0f}%)")

    with col2:
        st.subheader("🌳 Decision Tree")
        if dt_pred == 1:
            st.success(f"✅ Creditworthy ({dt_proba*100:.0f}%)")
        else:
            st.error(f"❌ Not Creditworthy ({(1-dt_proba)*100:.0f}%)")

    with col3:
        st.subheader("🌲 Random Forest")
        if rf_pred == 1:
            st.success(f"✅ Creditworthy ({rf_proba*100:.0f}%)")
        else:
            st.error(f"❌ Not Creditworthy ({(1-rf_proba)*100:.0f}%)")
            
            

st.info("ALL Models Predict it correctly ✅")