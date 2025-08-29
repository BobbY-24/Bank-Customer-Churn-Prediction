Bank Customer Churn Prediction
Project Overview
This project analyzes bank customer data to predict customer churn — when a customer closes their account. Accurately predicting churn helps banks take proactive measures to retain clients and improve business outcomes.
The project includes data exploration, visualization, PCA, feature correlation analysis, and predictive modeling using Random Forest classifiers.

Dataset
The dataset is from Kaggle: Churn Modelling Dataset and contains:
CreditScore


Geography


Gender


Age


Tenure


Balance


NumOfProducts


HasCrCard


IsActiveMember


EstimatedSalary


Exited (target variable: 1 = churned, 0 = stayed)



Key Steps
1. Data Exploration
Examined basic statistics and data types.


Visualized distributions of Age, CreditScore, Balance, and other numeric features.


Compared churn rates across Gender and Geography.


Generated a correlation heatmap for numeric features.


2. Insights from Correlation Analysis
Age vs. Exited: weak-to-moderate positive correlation (~0.29) → older customers slightly more likely to churn.


Number of Products vs. Exited: weak-to-moderate negative correlation (~–0.30) → customers with more products are slightly less likely to churn.


Other numeric features show weak correlations with churn.


Interpretation: Churn is influenced by multiple factors; no single variable dominates. Nonlinear models like Random Forest can capture interactions better than linear models.
3. Principal Component Analysis (PCA)
Standardized numeric features and reduced dimensions.


Explained variance of the first 2 components: ~23.7% → low, indicating variance is spread across many features.


Cumulative explained variance plot shows a roughly constant positive slope → each feature contributes similarly to variance.


Implication: PCA is useful for visualization but not ideal for compressing data without losing information. Random Forest handles this high-dimensional structure naturally.
4. Machine Learning Modeling
One-hot encoded categorical features.


Scaled numeric features using StandardScaler.


Trained a Random Forest Classifier for churn prediction.


Model Results on Test Set:


Accuracy: ~0.84


Precision, Recall, F1-score show:


High accuracy for non-churned customers


Moderate performance for predicting churned customers (common in imbalanced datasets)


Feature importance analysis revealed Age, Balance, and NumOfProducts as the most influential variables.


Interpretation: The Random Forest model performs well given weak linear correlations, leveraging nonlinear relationships and feature interactions to detect churn.
5. Visualization
Histograms, count plots, and heatmaps to explore data distributions and correlations.


PCA scatter plots for visualizing customer clusters.


Feature importance plots to interpret model decisions.



Usage
Clone the repository.


Place the dataset Churn_Modelling.csv in the working folder.


Run the Jupyter Notebook Churn_Prediction.ipynb.


Explore visualizations and model evaluation results.



Requirements
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
ipywidgets

Install using:
pip install pandas numpy matplotlib seaborn plotly scikit-learn ipywidgets


Key Insights
Older customers are slightly more likely to churn.


Customers with more products are slightly less likely to churn.


Variance in the dataset is distributed across many features (PCA slope ≈ constant).


Random Forest performs well because it captures nonlinear interactions among multiple variables, even when individual correlations are weak.


Feature engineering and interaction terms may further improve predictive performance.


Model results: ~84% test accuracy, good prediction for non-churned customers, moderate for churned customers.

