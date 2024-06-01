---
dataset_info:
  - config_name: default
    features:
      - name: Age
        dtype: int64 
      - name: Avg Monthly GB Download  
        dtype: float64
      - name: Avg Monthly Long Distance Charges
        dtype: float64
      - name: Churn Category 
        dtype: string 
      - name: Churn Label
        dtype: string  # Target Variable
      - name: Churn Reason
        dtype: string 
      - name: Churn Score
        dtype: int64
      - name: Churn Value
        dtype: int64
      - name: City
        dtype: string 
      - name: CLTV
        dtype: float64
      - name: Contract
        dtype: string 
      - name: Country
        dtype: string 
      - name: Customer ID
        dtype: string
      - name: Customer Status
        dtype: string 
      - name: Dependents
        dtype: bool
      - name: Device Protection Plan
        dtype: bool
      - name: Gender
        dtype: string 
      - name: Internet Service
        dtype: bool # Include this column
      - name: Internet Type
        dtype: string # Include this column
      - name: Lat Long
        dtype: string
      - name: Latitude
        dtype: float64
      - name: Longitude
        dtype: float64
      - name: Married
        dtype: bool
      - name: Monthly Charge
        dtype: float64
      - name: Multiple Lines 
        dtype: string 
      - name: Number of Dependents
        dtype: int64
      - name: Number of Referrals
        dtype: int64
      - name: Offer
        dtype: string 
      - name: Online Backup
        dtype: bool
      - name: Online Security
        dtype: bool
      - name: Paperless Billing
        dtype: bool
      - name: Partner
        dtype: bool # Include this column
      - name: Payment Method
        dtype: string 
      - name: Phone Service
        dtype: bool
      - name: Population
        dtype: int64
      - name: Premium Tech Support
        dtype: bool
      - name: Quarter
        dtype: string 
      - name: Referred a Friend
        dtype: bool
      - name: Satisfaction Score
        dtype: int64  
      - name: Senior Citizen  
        dtype: bool
      - name: State  
        dtype: string 
      - name: Streaming Movies 
        dtype: bool
      - name: Streaming Music
        dtype: bool
      - name: Streaming TV
        dtype: bool
      - name: Tenure in Months
        dtype: int64
      - name: Total Charges
        dtype: float64
      - name: Total Extra Data Charges
        dtype: float64
      - name: Total Long Distance Charges
        dtype: float64
      - name: Total Refunds
        dtype: float64
      - name: Total Revenue
        dtype: float64
      - name: Under 30
        dtype: bool
      - name: Unlimited Data
        dtype: bool
      - name: Zip Code
        dtype: string
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