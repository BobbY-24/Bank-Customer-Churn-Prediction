# Bank Customer Churn Prediction

## Overview
I built a supervised machine learning pipeline to predict whether a bank customer is likely to churn. It combines exploratory data analysis, feature preprocessing, dimensionality exploration with PCA, and Random Forest classification. My goal is to understand which customer attributes are associated with churn and evaluate how well a nonlinear model can identify at-risk customers.

## Motivation
Customer churn prediction is a practical classification problem with business and statistical relevance. For a data science portfolio, this project demonstrates end-to-end work with tabular data: cleaning, visualization, categorical encoding, model training, evaluation, and interpretation. It also highlights an important modeling lesson: high overall accuracy can hide weaker performance on the minority class.

## Dataset
- **Source:** Kaggle Churn Modelling dataset.
- **File:** `data/Churn_Modelling.csv`
- **Size:** 10,000 customer records and 14 columns.
- **Target variable:** `Exited`, where `1` indicates the customer churned and `0` indicates the customer stayed.
- **Important features:** `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, and `EstimatedSalary`.
- **Known limitations:** The dataset is a simplified public benchmark. It may not represent real bank populations, time-varying customer behavior, intervention effects, or production data drift.

## Methods
- Loaded and inspected the customer churn dataset with pandas.
- Explored numeric and categorical feature distributions using matplotlib, seaborn, and Plotly.
- Examined churn rates across demographic and account-related variables.
- One-hot encoded categorical features and scaled numeric features.
- Used PCA to inspect variance structure and determine whether dimensionality reduction was informative.
- Trained a Random Forest classifier for churn prediction.
- Evaluated the model with accuracy, precision, recall, F1-score, and a classification report.

## Results
The Random Forest model achieved approximately **85.7% test accuracy**.

Classification report from the notebook:

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| Stayed (`0`) | 0.86 | 0.98 | 0.92 | 1607 |
| Churned (`1`) | 0.84 | 0.34 | 0.48 | 393 |

The model performs much better on non-churned customers than churned customers, which is expected for an imbalanced churn dataset. This makes recall for churned customers a key improvement target.

## Key Insights
- Age has a weak-to-moderate positive relationship with churn in this dataset.
- Customers with more products are less likely to churn, based on the observed correlation and model importance.
- PCA explained limited variance in the first two components, suggesting that the signal is distributed across multiple features.
- Random Forest can capture nonlinear interactions that are not visible in simple correlation analysis.
- Accuracy alone is not sufficient for churn modeling because the minority churn class is harder to detect.

## Limitations
- I use a single train/test split rather than cross-validation.
- I do not yet tune hyperparameters or compare several model families.
- The dataset does not include temporal customer history, marketing interventions, or economic context.
- The model is descriptive and predictive; it does not prove causal drivers of churn.
- Minority-class recall is modest, so the model would need additional work before operational use.

## Future Improvements
- Add cross-validation and confidence intervals for model metrics.
- Tune Random Forest hyperparameters and compare with logistic regression, XGBoost, or LightGBM.
- Add class weighting, threshold tuning, or resampling to improve churn recall.
- Create a compact `src/` training script after the notebook workflow is finalized.
- Add a short model card describing intended use and limitations.

## How to Run
```bash
git clone https://github.com/BobbY-24/Bank-Customer-Churn-Prediction.git
cd Bank-Customer-Churn-Prediction
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
jupyter notebook notebooks/bank_customer_churn_prediction.ipynb
```

Run my notebook cells from top to bottom. I expect the dataset at `data/Churn_Modelling.csv`.
