# CreditWise — Loan Approval Prediction System

A machine learning project that predicts whether a bank loan application should be **Approved** or **Rejected**, based on applicant financial and demographic data. Built as a final year project for **SecureTrust Bank's** proposed intelligent loan screening system.

---

## 1. Problem Statement

SecureTrust Bank currently approves or rejects loans through manual document verification — checking income proofs, employment details, and credit history by hand. This process is slow and inconsistent, and produces two costly types of errors:

1. **Good customers get rejected** → lost business.
2. **High-risk customers get approved** → financial losses.

The goal of this project is to build a model that learns patterns from historical loan applications and flags applications as likely **Approved** or **Rejected** before final human review — reducing both error types.

---

## 2. Dataset

- **Raw size:** 1,000 loan applications, 20 columns (19 features + 1 target)
- **Missing data:** every column has ~50 missing values (5%)
- **Working size:** 950 applications after dropping the 50 rows with a missing target (see Section 3.1 — labels are never imputed)
- **Target distribution:** ~69% Rejected, ~31% Approved (moderately imbalanced)

| Column | Description |
|---|---|
| Applicant_ID | Unique applicant ID |
| Applicant_Income | Monthly income of applicant |
| Coapplicant_Income | Monthly income of co-applicant |
| Employment_Status | Salaried / Self-Employed / Business |
| Age | Applicant age |
| Marital_Status | Married / Single |
| Dependents | Number of dependents |
| Credit_Score | Credit bureau score |
| Existing_Loans | Number of already running loans |
| DTI_Ratio | Debt-to-Income ratio |
| Savings | Savings balance |
| Collateral_Value | Value of collateral provided |
| Loan_Amount | Loan amount requested |
| Loan_Term | Loan duration (months) |
| Loan_Purpose | Home / Education / Personal / Business |
| Property_Area | Urban / Semi-Urban / Rural |
| Education_Level | Graduate / Postgraduate / Undergraduate |
| Gender | Male / Female |
| Employer_Category | Govt / Private / Self |
| **Loan_Approved** | **Target: 1 = Approved, 0 = Rejected** |

---

## 3. Workflow

### 3.1 Data Cleaning
- Rows with a **missing target** (`Loan_Approved`) are dropped outright — 50 rows (5%), leaving 950. A label is either known or the row isn't used for training; it is never guessed by an imputer.
- Missing numerical values in the remaining rows filled with the **mean** (`SimpleImputer(strategy="mean")`).
- Missing categorical values filled with the **mode** (`SimpleImputer(strategy="most_frequent")`), explicitly excluding the target column.
- `Applicant_ID` dropped (identifier, not predictive).

### 3.2 Exploratory Data Analysis
- Class balance check (pie chart)
- Distribution plots for income, co-applicant income, credit score
- Boxplots of income, credit score, DTI ratio, and savings split by approval outcome
- Correlation heatmap across all engineered features

**Key EDA findings:**
- **Credit_Score** is the strongest single predictor of approval (correlation ≈ **0.45**)
- **DTI_Ratio** is the strongest negative predictor (correlation ≈ **-0.44**) — higher debt-to-income strongly associates with rejection
- **Applicant_Income** alone is a weak predictor (correlation ≈ 0.12) — income matters far less than how much of it is already committed to debt
- Most other features (employment type, property area, gender, education) show little to no correlation with approval

### 3.3 Feature Engineering
- Label encoding for `Education_Level` and the target
- One-hot encoding (drop-first) for `Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Gender`, `Employer_Category`
- Squared terms added: `DTI_Ratio_sq`, `Credit_Score_sq` (second pass, replacing the raw linear terms)
- `StandardScaler` applied to all features before model training

### 3.4 Models Trained
Three classifiers were trained and compared, once on the base feature set and once after feature engineering:

- Logistic Regression
- K-Nearest Neighbors (k=5)
- Gaussian Naive Bayes

---

## 4. Results

*(Metrics below are computed on the cleaned 950-row dataset, after dropping rows with a missing target — see Section 3.1.)*

### Baseline features (before squared terms)

| Model | Precision | Recall | F1 Score | Accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.873 | 0.787 | 0.828 | 0.895 |
| KNN (k=5) | 0.733 | 0.541 | 0.623 | 0.789 |
| Naive Bayes | 0.898 | 0.721 | 0.800 | 0.884 |

### After feature engineering (DTI_Ratio_sq, Credit_Score_sq)

| Model | Precision | Recall | F1 Score | Accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.860 | **0.803** | **0.831** | 0.895 |
| KNN (k=5) | 0.744 | 0.525 | 0.615 | 0.789 |
| Naive Bayes | **0.902** | 0.754 | 0.821 | 0.895 |

**Takeaway:** Naive Bayes has the highest precision in both rounds. Logistic Regression has the highest recall and F1, and ties Naive Bayes on accuracy after feature engineering. Which model is "best" genuinely depends on what the bank wants to optimize for:
- Optimizing for **precision** (don't approve risky applicants) → Naive Bayes
- Optimizing for **recall** and overall balance (don't reject good applicants, catch more approvals correctly) → Logistic Regression

This project reports both rather than declaring a single winner, since the trade-off is a business decision, not a purely technical one.

---

## 5. Tech Stack

- Python 3
- pandas, numpy — data handling
- seaborn, matplotlib — visualization
- scikit-learn — preprocessing, modeling, evaluation
- Jupyter Notebook — development environment

---

## 6. Project Structure

```
CreditWise-Loan-Approval-Prediction/
│
├── 📂 data/
│   └── loan_approval_data.csv
│
├── 📂 notebook/
│   └── credit_wise.ipynb
│
├── 📂 images/
│   ├── applicant_income_distribution.png
│   ├── applicant_income_by_loan_status.png
│   ├── applicant_income_vs_loan_approval.png
│   ├── coapplicant_income_distribution.png
│   ├── credit_score_vs_loan_approval.png
│   ├── education_level_distribution.png
│   ├── feature_analysis_boxplots.png
│   ├── feature_correlation_heatmap.png
│   └── loan_approval_distribution.png
│
├── 📂 problem_statement/
│   ├── Problem Statement (Part-1).png
│   └── Dataset Description.png
│
├── 📄 requirements.txt
├── 📄 LICENSE
├── 📄 .gitignore
└── 📄 README.md
```

---

## 7. How to Run

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd CreditWise-Loan-Approval-Prediction

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook notebook/credit_wise.ipynb
```

---

## 8. Limitations & Next Steps

Being upfront about what this project doesn't yet do:

- **No cross-validation.** Results come from a single 80/20 split (~200 test rows). K-fold cross-validation would give a more reliable estimate of how these models generalize.
- **No hyperparameter tuning.** All three models use scikit-learn defaults. A GridSearch/RandomizedSearch pass, especially on KNN's `k` and Logistic Regression's regularization strength, is a natural next step.
- **No tree-based or ensemble models yet.** Random Forest and Gradient Boosting (e.g. XGBoost) typically handle this kind of tabular, mixed-type data better than linear/instance-based models and are worth benchmarking next.
- **Class imbalance (69/31) is measured, not corrected.** Precision/recall/F1 are reported (good — better than accuracy alone), but no resampling (SMOTE) or `class_weight='balanced'` has been applied yet.
- **No deployment layer.** This is currently an analysis notebook, not a runnable service — no saved model file, no API, no UI. A `joblib`-exported model behind a simple Flask/FastAPI endpoint would be the logical next step toward an actual "system."

---

## 9. Author

**Vivek Kumar**
Final Year Project — Machine Learning

- LinkedIn: [linkedin.com/in/vivek-kumar-66041b410](https://www.linkedin.com/in/vivek-kumar-66041b410)
- GitHub: [github.com/vivekkr-data](https://github.com/vivekkr-data)
