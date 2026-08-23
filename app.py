"""Streamlit interface for the CreditWise decision-support demo."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.model import feature_importance, load_data, train_production_model


st.set_page_config(
    page_title="CreditWise Loan System",
    page_icon="💳",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem;}
    [data-testid="stMetricValue"] {color: #0f766e;}
    .result-card {padding: 1.1rem 1.3rem; border-radius: 12px; background: #f0fdfa;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Training the CreditWise model...")
def get_model():
    features, target = load_data()
    return train_production_model(features, target), len(target)


model, training_rows = get_model()

st.title("CreditWise — Loan Approval Prediction")
st.caption(
    "Machine-learning decision-support demo. Predictions are estimates and must "
    "not replace affordability checks, policy rules, or human review."
)

with st.sidebar:
    st.header("Model card")
    st.metric("Labelled applications", f"{training_rows:,}")
    st.metric("Model", "Gradient Boosting")
    st.metric("Verified holdout accuracy", "96.32%")
    st.info(
        "Applicant ID, gender, and marital status are not used by the prediction model."
    )

st.subheader("Applicant and loan details")
first, second, third = st.columns(3)

with first:
    applicant_income = st.number_input("Applicant monthly income", 0, 1_000_000, 10_500)
    coapplicant_income = st.number_input("Co-applicant monthly income", 0, 1_000_000, 5_000)
    employment_status = st.selectbox(
        "Employment status", ["Salaried", "Self-employed", "Contract", "Unemployed"]
    )
    age = st.number_input("Age", 18, 100, 40)
    dependents = st.number_input("Dependents", 0, 20, 1)
    credit_score = st.number_input("Credit score", 300, 900, 680)

with second:
    existing_loans = st.number_input("Existing loans", 0, 20, 1)
    dti_ratio = st.slider("Debt-to-income ratio", 0.0, 1.0, 0.35, 0.01)
    savings = st.number_input("Savings balance", 0, 10_000_000, 10_000)
    collateral_value = st.number_input("Collateral value", 0, 10_000_000, 25_000)
    loan_amount = st.number_input("Requested loan amount", 0, 10_000_000, 20_000)
    loan_term = st.selectbox("Loan term (months)", [12, 24, 36, 48, 60, 72, 84], index=3)

with third:
    loan_purpose = st.selectbox(
        "Loan purpose", ["Home", "Education", "Personal", "Business", "Car"]
    )
    property_area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])
    education_level = st.selectbox("Education level", ["Graduate", "Not Graduate"])
    employer_category = st.selectbox(
        "Employer category", ["Private", "Government", "MNC", "Business", "Unemployed"]
    )

application = pd.DataFrame(
    [
        {
            "Applicant_Income": applicant_income,
            "Coapplicant_Income": coapplicant_income,
            "Employment_Status": employment_status,
            "Age": age,
            "Dependents": dependents,
            "Credit_Score": credit_score,
            "Existing_Loans": existing_loans,
            "DTI_Ratio": dti_ratio,
            "Savings": savings,
            "Collateral_Value": collateral_value,
            "Loan_Amount": loan_amount,
            "Loan_Term": loan_term,
            "Loan_Purpose": loan_purpose,
            "Property_Area": property_area,
            "Education_Level": education_level,
            "Employer_Category": employer_category,
        }
    ]
)

if st.button("Estimate approval likelihood", type="primary", width="stretch"):
    probability = float(model.predict_proba(application)[0, 1])
    predicted_approval = probability >= 0.50

    st.divider()
    left, right = st.columns([2, 1])
    with left:
        if predicted_approval:
            st.success("Model result: likely to be approved")
        else:
            st.warning("Model result: likely to be rejected")
        st.progress(probability, text=f"Estimated approval probability: {probability:.1%}")
        st.caption(
            "A 50% demonstration threshold is used. A real lender must validate and "
            "set its threshold according to risk policy, regulation, and fairness testing."
        )
    with right:
        st.metric("Approval probability", f"{probability:.1%}")
        st.metric("Decision threshold", "50%")

with st.expander("What influences this model?"):
    st.dataframe(
        feature_importance(model).rename(
            columns={"feature": "Transformed feature", "importance": "Importance"}
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Feature importance describes this fitted model, not a causal relationship."
    )
