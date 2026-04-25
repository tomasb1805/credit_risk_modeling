Credit Risk Modeling and Underwriting Automation

## 1. Abstract

This project establishes an automated pipeline for predicting loan default probabilities using machine learning. It integrates traditional demographic and loan application data with synthetic transactional data in open-banking format, sourced via the Plaid API. The pipeline encompasses data extraction, feature engineering, XGBoost model optimization, and SHAP-based interpretability to support automated, transparent underwriting decisions.

## 2. Data Architecture and ETL

The data ingestion and transformation framework utilizes Python and PostgreSQL.

#### Static Loan Data (etl.py): 
Ingests raw loan application records. Processes include snake_case column standardization, binary feature encoding, and heuristic-based risk persona assignment (e.g., gig_worker, financially_stretched, broad_income_mix).

####  Open-Banking Data (plaid_python_api_client.py & plaid_data_etl.py):
Utilizes the Plaid Sandbox API to extract synthetic banking accounts and transactions mapped to the defined personas.

#### Database Schema:
Cleaned data is loaded into a PostgreSQL database. Application data resides in raw.loan_applications, transactional data in raw.transactions, and derived metrics in analytics.persona_features. A SQL join integrates loan application data and behavioral features for modeling.

## 3. Feature Engineering

Feature derivation is conducted in two primary phases:

#### Behavioral Statistics (statistical_features_derivation.py):
Transactional data is aggregated at the persona level to quantify cash flow health. Key metrics include:

- avg_monthly_inflow

- income_stability_ratio

- spend_to_income_ratio

- months_net_negative_6m

- cash_withdrawal_ratio

#### Interaction Features (build_model_dataset.py):
Financial ratios are engineered to capture borrower leverage and stability, including:
- debt_to_income
- rate_per_term
- credit_utilisation
- employment_stability
[being computed as: $\ln(1 + \text{income} \times \text{months-employed})$]

Missing transactional data for the default mainstream persona is addressed via explicit flag indicators (_missing flags) to preserve structural information.

## 4. Modeling Strategy

The predictive model is built using XGBClassifier (train_xgb_model.py).

#### Preprocessing: 
A Scikit-Learn ColumnTransformer applies median imputation to continuous variables through SimpleImputer and categorical variables using One-Hot Encoding and mode imputation.

#### Imbalance Handling:
Class imbalance is mitigated by scaling the positive class weights dynamically based on the training set distribution.

#### Hyperparameter Optimization:
Model tuning is executed via RandomizedSearchCV using 5-fold cross-validation. The objective function maximizes the ROC-AUC score across an extensive grid of tree depth, learning rate, and regularization parameters.

## 5. Model Evaluation

Model performance is evaluated out-of-sample (evaluate_model.py).

#### ROC-AUC:
Quantifies the model's discriminative capacity between default and non-default states across all classification thresholds.

#### Precision-Recall (Average Precision):
Utilized to evaluate performance specific to the minority class (defaulters), providing a more rigorous assessment of false positive vs. false negative trade-offs.

## 6. Model Interpretability and Underwriting

To comply with financial interpretability and compliance requirements, SHAP (SHapley Additive exPlanations) is integrated in the model pipeline (explain_model.py).

#### Global Interpretability:
Beeswarm summary plots and dependence plots identify macro-level drivers of credit risk.

#### Local Interpretability:
Waterfall plots disaggregate individual prediction probabilities into base values and specific feature contributions.

#### Automated Decision policy:
Probabilities are mapped to discrete underwriting actions based on risk tolerance thresholds:

$P(\text{Default}) < 0.20$: Auto-Approve

$0.20 \leq P(\text{Default}) \leq 0.35$: Refer for Manual Review

$P(\text{Default}) > 0.35$: Auto-Decline

## 7. Conclusion

The deployed architecture successfully integrates heterogeneous data sources to generate robust credit risk predictions. The use of an optimized gradient boosting framework combined with SHAP ensures the underwriting process is both highly predictive and strictly interpretable.
