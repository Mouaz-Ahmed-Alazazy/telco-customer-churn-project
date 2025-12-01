# Customer Churn Prediction Using Machine Learning

**Abstract**
This project aims to enhance customer retention in the telecommunications industry by developing machine learning models to predict customer churn. Using a dataset from IBM (via Kaggle) representing a fictional telco company in California, we analyze customer demographics, services, and account information to identify key drivers of churn. This midway report outlines the problem motivation, methodology, preliminary data analysis, and the proposed experimental design, which includes handling class imbalance with SMOTE and evaluating models using cost-sensitive learning techniques.

## 1. Introduction
### 1.1 Problem Statement
Customer churn, or customer attrition, refers to the phenomenon where customers stop doing business with a company. In the highly competitive telecommunications sector, churn is a critical challenge. The cost of acquiring a new customer is significantly higher—often 5 to 25 times more—than retaining an existing one. Therefore, predicting which customers are at risk of leaving is essential for proactive retention strategies.

### 1.2 Motivation
While traditional methods of churn analysis rely on historical reporting and manual intuition, they often fail to capture complex, non-linear patterns in customer behavior. Machine Learning (ML) offers a scalable and automated approach to predict churn probability at the individual level. By leveraging ML, companies can identify at-risk customers early and deploy targeted interventions, thereby reducing revenue loss and improving customer lifetime value.

### 1.3 Objectives
The primary objective of this project is to build robust classification models (Logistic Regression, Random Forest, XGBoost) to predict customer churn. Beyond standard accuracy, this project aims to:
*   Address the inherent class imbalance problem using Synthetic Minority Over-sampling Technique (SMOTE).
*   Evaluate the impact of different cost functions (e.g., Log Loss vs. Focal Loss) to penalize false negatives more heavily, reflecting the high cost of losing a customer.
*   Compare hyperparameter tuning strategies (Grid Search vs. Random Search) to optimize model performance.

## 2. Methodology
### 2.1 Data Description
The dataset used in this study is the "Telco Customer Churn" dataset, originally provided by IBM Sample Data Sets and hosted on Kaggle. It represents a fictional telecommunications company providing home phone and internet services to 7,043 customers in California.
*   **Volume**: The training subset contains 4,227 observations.
*   **Features**: There are 52 columns (after initial splitting/formatting), including:
    *   **Demographics**: Age, Gender, Partner, Dependents.
    *   **Services**: Phone Service, Internet Service, Online Security, Streaming, etc.
    *   **Account Information**: Contract type, Payment Method, Monthly Charges, Tenure.
*   **Target Variable**: `Churn` (Binary: 1 for Churn, 0 for Stayed).

### 2.2 Preprocessing
*   **Missing Values**: Missing values in `Churn Category` and `Churn Reason` were identified as present only for non-churners. These are handled by imputing a "No Churn" category or excluding them from the feature set for prediction.
*   **Encoding**: Categorical variables such as `Gender` and `Partner` are label-encoded, while multi-category features like `Payment Method` are one-hot encoded.
*   **Scaling**: Numerical features like `Tenure` and `Monthly Charges` are standardized to ensure model convergence.
*   **Class Imbalance Handling**: The dataset exhibits a class imbalance with approximately 27% churners. To prevent model bias, we employ **SMOTE (Synthetic Minority Over-sampling Technique)** during the training phase.

### 2.3 Model Selection and Experimental Design
We are implementing and comparing three distinct algorithms:
1.  **Logistic Regression**: As a baseline for interpretability.
2.  **Random Forest**: To capture non-linear relationships and feature importance.
3.  **XGBoost**: For high predictive performance and gradient boosting capabilities.

**Experimental Goals**:
*   **Cost-Sensitive Learning**: We are experimenting with custom cost functions that assign a higher penalty to False Negatives compared to False Positives.
*   **Hyperparameter Tuning**: We compare the efficiency of Grid Search versus Random Search for optimizing model parameters.
*   **Churn Timing**: We analyze churn rates across different tenure cohorts to identify "timing patterns".
## 3. Preliminary Results
Exploratory Data Analysis (EDA) on the training set reveals several key insights:
*   **Churn Distribution**: The dataset is imbalanced, confirming the need for resampling techniques.
*   **Contract Type**: Customers with "Month-to-Month" contracts have a significantly higher churn rate compared to those with "One Year" or "Two Year" contracts.
*   **Tenure**: There is a strong negative correlation between tenure and churn; newer customers are more likely to leave.
*   **Monthly Charges**: Higher monthly charges are associated with a slightly higher likelihood of churn.

## 4. Risk Analysis
*   **Data Bias**: Since the dataset is synthetic/fictional (IBM sample data), it may not perfectly reflect the complexities of real-world human behavior or economic shifts.
*   **Overfitting**: Complex models like XGBoost are prone to overfitting, especially with oversampled data. This will be mitigated using Stratified K-Fold Cross-Validation.
*   **Implementation Complexity**: Implementing custom cost functions and ensuring they converge correctly can be technically challenging and may require extensive tuning.

## 5. Future Work
The next phase of the project will focus on:
1.  Executing the preprocessing pipeline, including SMOTE.
2.  Training the selected models and performing hyperparameter tuning.
3.  Evaluating models using F1-Score, AUC-ROC, and a custom cost-sensitive accuracy metric.
4.  Finalizing the report with comparative results and business recommendations.

## 6. References
[1] IBM Sample Data Sets, "Telco Customer Churn," Kaggle. [Online]. Available: https://www.kaggle.com/blastchar/telco-customer-churn.
[2] N. V. Chawla, et al., "SMOTE: Synthetic Minority Over-sampling Technique," J. Artif. Intell. Res., vol. 16, pp. 321–357, 2002.
[3] C. M. Bishop, *Pattern Recognition and Machine Learning*. Springer, 2006.
