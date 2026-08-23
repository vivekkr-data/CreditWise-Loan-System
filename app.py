"""Streamlit interface for the CreditWise decision-support demo."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.model import (
    CATEGORICAL_FEATURES,
    feature_importance,
    load_data,
    numeric_feature_ranges,
    train_production_model,
    validate_numeric_ranges,
)

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
    categories = {
        column: sorted(features[column].dropna().astype(str).unique().tolist())
        for column in CATEGORICAL_FEATURES
    }
    loan_terms = sorted(
        int(value) for value in features["Loan_Term"].dropna().unique().tolist()
    )
    return (
        train_production_model(features, target),
        len(target),
        numeric_feature_ranges(features),
        categories,
        loan_terms,
    )


model, training_rows, observed_ranges, category_options, loan_terms = get_model()


def integer_range(column: str) -> tuple[int, int]:
    """Return an integer-safe observed range for a Streamlit input."""
    minimum, maximum = observed_ranges[column]
    return int(minimum), int(maximum)


def option_index(column: str, preferred: str) -> int:
    """Use a preferred default when it exists in the dataset."""
    options = category_options[column]
    return options.index(preferred) if preferred in options else 0


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
    st.info("Applicant ID, age, gender, and marital status are not used by the model.")

st.subheader("Applicant and loan details")
first, second, third = st.columns(3)

with first:
    applicant_income = st.number_input(
        "Applicant monthly income", *integer_range("Applicant_Income"), 10_500
    )
    coapplicant_income = st.number_input(
        "Co-applicant monthly income", *integer_range("Coapplicant_Income"), 5_000
    )
    employment_status = st.selectbox(
        "Employment status",
        category_options["Employment_Status"],
        index=option_index("Employment_Status", "Salaried"),
    )
    dependents = st.number_input("Dependents", *integer_range("Dependents"), 1)
    credit_score = st.number_input("Credit score", *integer_range("Credit_Score"), 680)

with second:
    existing_loans = st.number_input(
        "Existing loans", *integer_range("Existing_Loans"), 1
    )
    dti_ratio = st.slider(
        "Debt-to-income ratio",
        *observed_ranges["DTI_Ratio"],
        0.35,
        0.01,
    )
    savings = st.number_input("Savings balance", *integer_range("Savings"), 10_000)
    collateral_value = st.number_input(
        "Collateral value", *integer_range("Collateral_Value"), 25_000
    )
    loan_amount = st.number_input(
        "Requested loan amount", *integer_range("Loan_Amount"), 20_000
    )
    loan_term = st.selectbox(
        "Loan term (months)",
        loan_terms,
        index=loan_terms.index(48) if 48 in loan_terms else 0,
    )

with third:
    loan_purpose = st.selectbox(
        "Loan purpose",
        category_options["Loan_Purpose"],
        index=option_index("Loan_Purpose", "Home"),
    )
    property_area = st.selectbox(
        "Property area",
        category_options["Property_Area"],
        index=option_index("Property_Area", "Urban"),
    )
    education_level = st.selectbox(
        "Education level",
        category_options["Education_Level"],
        index=option_index("Education_Level", "Graduate"),
    )
    employer_category = st.selectbox(
        "Employer category",
        category_options["Employer_Category"],
        index=option_index("Employer_Category", "Private"),
    )

st.caption(
    "Input controls are limited to values observed in the bundled training data. "
    "This prevents unsupported out-of-distribution estimates."
)

application = pd.DataFrame(
    [
        {
            "Applicant_Income": applicant_income,
            "Coapplicant_Income": coapplicant_income,
            "Employment_Status": employment_status,
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
    range_errors = validate_numeric_ranges(application, observed_ranges)
    if range_errors:
        st.error("Cannot score this application: " + "; ".join(range_errors))
    else:
        model_score = float(model.predict_proba(application)[0, 1])
        predicted_approval = model_score >= 0.50

        st.divider()
        left, right = st.columns([2, 1])
        with left:
            if predicted_approval:
                st.success("Model result: likely to be approved")
            else:
                st.warning("Model result: likely to be rejected")
            st.progress(model_score, text=f"Model approval score: {model_score:.1%}")
            st.caption(
                "This is an uncalibrated model score, not a guaranteed real-world "
                "probability. A 50% demonstration threshold is used."
            )
        with right:
            st.metric("Model approval score", f"{model_score:.1%}")
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
