---
language:
- en
tags:
- tabular-classification
- churn-prediction
- telecom
- customer-retention
- demographics
- customer-service
pretty_name: Telco Customer Churn
dataset_info:
  - config_name: default
    features:
      - name: Customer ID
        dtype: string
      - name: Gender
        dtype: string 
      - name: Age
        dtype: int64 
      - name: Under 30
        dtype: bool
      - name: Senior Citizen  
        dtype: bool
      - name: Married
        dtype: bool
      - name: Dependents
        dtype: bool
      - name: Number of Dependents
        dtype: int64
      - name: Country
        dtype: string 
      - name: State  
        dtype: string 
      - name: City
        dtype: string 
      - name: Zip Code
        dtype: string 
      - name: Lat Long
        dtype: string
      - name: Latitude
        dtype: float64
      - name: Longitude
        dtype: float64
      - name: Population
        dtype: int64
      - name: Quarter
        dtype: string 
      - name: Referred a Friend
        dtype: bool
      - name: Number of Referrals
        dtype: int64
      - name: Tenure in Months
        dtype: int64
      - name: Offer
        dtype: string 
      - name: Phone Service
        dtype: bool
      - name: Avg Monthly Long Distance Charges
        dtype: float64
      - name: Multiple Lines
        dtype: string 
      - name: Internet Service
        dtype: bool
      - name: Internet Type
        dtype: string 
      - name: Avg Monthly GB Download  
        dtype: float64
      - name: Online Security
        dtype: bool
      - name: Online Backup
        dtype: bool
      - name: Device Protection Plan
        dtype: bool
      - name: Premium Tech Support
        dtype: bool
      - name: Streaming TV
        dtype: bool
      - name: Streaming Movies 
        dtype: bool
      - name: Streaming Music
        dtype: bool
      - name: Unlimited Data
        dtype: bool
      - name: Contract
        dtype: string 
      - name: Paperless Billing
        dtype: bool
      - name: Payment Method
        dtype: string 
      - name: Monthly Charge
        dtype: float64
      - name: Total Charges
        dtype: float64
      - name: Total Refunds
        dtype: float64
      - name: Total Extra Data Charges
        dtype: float64
      - name: Total Long Distance Charges
        dtype: float64
      - name: Total Revenue
        dtype: float64
      - name: Satisfaction Score
        dtype: int64  
      - name: Customer Status
        dtype: string 
      - name: Churn Label
        dtype: string 
      - name: Churn Value
        dtype: int64
      - name: Churn Score
        dtype: int64
      - name: CLTV
        dtype: float64
      - name: Churn Category 
        dtype: string 
      - name: Churn Reason
        dtype: string 
      - name: Partner
        dtype: bool
configs:
- config_name: default
  data_files:
  - split: train
    path: train.csv
  - split: test
    path: test.csv
train-eval-index:
- config: default
  task: text-classification
  task_id: multi_label_classification 
  col_mapping:
    Customer ID: id
    Gender: Gender
    Age: Age
    Under 30: Under 30
    Senior Citizen: Senior Citizen
    Married: Married
    Dependents: Dependents
    Number of Dependents: Number of Dependents
    Country: Country
    State: State
    City: City
    Zip Code: Zip Code
    Lat Long: Lat Long
    Latitude: Latitude
    Longitude: Longitude
    Population: Population
    Quarter: Quarter
    Referred a Friend: Referred a Friend
    Number of Referrals: Number of Referrals
    Tenure in Months: Tenure in Months
    Offer: Offer
    Phone Service: Phone Service
    Avg Monthly Long Distance Charges: Avg Monthly Long Distance Charges
    Multiple Lines: Multiple Lines
    Internet Service: Internet Service
    Internet Type: Internet Type
    Avg Monthly GB Download: Avg Monthly GB Download
    Online Security: Online Security
    Online Backup: Online Backup
    Device Protection Plan: Device Protection Plan
    Premium Tech Support: Premium Tech Support
    Streaming TV: Streaming TV
    Streaming Movies: Streaming Movies
    Streaming Music: Streaming Music
    Unlimited Data: Unlimited Data
    Contract: Contract
    Paperless Billing: Paperless Billing
    Payment Method: Payment Method
    Monthly Charge: Monthly Charge
    Total Charges: Total Charges
    Total Refunds: Total Refunds
    Total Extra Data Charges: Total Extra Data Charges
    Total Long Distance Charges: Total Long Distance Charges
    Total Revenue: Total Revenue
    Satisfaction Score: Satisfaction Score
    Customer Status: Customer Status
    Churn Label: label
    Churn Value: Churn Value
    Churn Score: Churn Score
    CLTV: CLTV
    Churn Category: Churn Category
    Churn Reason: Churn Reason
    Partner: Partner
  metrics:
  - type: accuracy
    name: Accuracy
  - type: precision
    name: Precision
  - type: recall
    name: Recall
  - type: f1
    name: F1 Score
---
## Telco Customer Churn

**This dataset is a valuable resource for exploring and predicting customer churn in the telecommunications industry. It provides a comprehensive snapshot of customer demographics, service usage patterns, billing information, and churn status, making it ideal for training machine learning models to predict customer churn and develop effective customer retention strategies.**

**Content and Structure:**

The dataset is structured in a tabular format, with each row representing a unique customer and each column containing attributes about that customer.  

* **Customer Demographics:**  Features like gender, age, marital status, and dependents provide insights into customer profiles.
* **Service Usage:**  Details customer subscriptions to services such as phone, internet, multiple lines, online security, online backup, device protection, tech support, and streaming options.
* **Billing Information:**  Provides data on tenure, contract type, payment method, monthly charges, and total charges.
* **Churn Information:**  Includes labels indicating whether a customer churned, the reason for churn (if applicable), and churn scores for analysis.

**Data Collection and Curation:**

This dataset is a fictional dataset created by IBM data scientists as a sample dataset for exploring customer churn prediction. It is not based on real-world data and should be treated as a simulation for learning and experimentation.

**Usage Examples:**

* **Customer Churn Prediction:** Train classification models to predict churn based on customer demographics, service usage, and billing information.
* **Customer Segmentation:** Analyze the dataset to identify customer segments with different churn probabilities, allowing for targeted retention strategies.
* **Feature Engineering:** Experiment with feature engineering techniques to improve churn prediction model accuracy.

**Additional Information:**

* **Industry Relevance:**  Relevant for businesses in the telecommunications industry and other sectors that deal with customer churn. 
* **Ethical Considerations:**  This is a fictional dataset and does not contain real personal or sensitive information. 