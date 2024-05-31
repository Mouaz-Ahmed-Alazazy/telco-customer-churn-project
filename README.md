language:
- en
license: other
license_name: Telco Customer Churn License
license_link: LICENSE
license_details: This dataset is provided under the Telco Customer Churn License, which allows for free use and distribution for non-commercial purposes. Commercial use requires explicit permission from the creators.
tags:
- tabular-classification
- churn-prediction
- telecom
- customer-retention
- demographics
- customer-service
annotations_creators:
- no-annotation
language_creators:
- found
language_details:
- en
pretty_name: Telco Customer Churn
size_categories:
- 10K<n<100K
source_datasets:
- original
task_categories:
- tabular-classification
task_ids:
- multi-class-classification
paperswithcode_id: null
configs:
- config_name: default
  data_files:
  - split: train
    path: telco_customer_churn.csv
dataset_info:
  features:
    - name: Customer ID
      dtype: string
    - name: Gender
      dtype: string
    - name: Age
      dtype: int32
    - name: Under 30
      dtype: bool
    - name: Senior Citizen
      dtype: bool
    - name: Married
      dtype: bool
    - name: Dependents
      dtype: bool
    - name: Number of Dependents
      dtype: int32
    - name: Country
      dtype: string
    - name: State
      dtype: string
    - name: City
      dtype: string
    - name: Zip Code
      dtype: int32
    - name: Lat Long
      dtype: string
    - name: Latitude
      dtype: float32
    - name: Longitude
      dtype: float32
    - name: Population
      dtype: int32
    - name: Quarter
      dtype: string
    - name: Referred a Friend
      dtype: bool
    - name: Number of Referrals
      dtype: int32
    - name: Tenure in Months
      dtype: int32
    - name: Offer
      dtype: string
    - name: Phone Service
      dtype: bool
    - name: Avg Monthly Long Distance Charges
      dtype: float32
    - name: Multiple Lines
      dtype: bool
    - name: Internet Service
      dtype: bool
    - name: Internet Type
      dtype: string
    - name: Avg Monthly GB Download
      dtype: float32
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
      dtype: float32
    - name: Total Charges
      dtype: float32
    - name: Total Refunds
      dtype: float32
    - name: Total Extra Data Charges
      dtype: float32
    - name: Total Long Distance Charges
      dtype: float32
    - name: Total Revenue
      dtype: float32
    - name: Satisfaction Score
      dtype: int32
    - name: Customer Status
      dtype: string
    - name: Churn Label
      dtype: string
    - name: Churn Value
      dtype: bool
    - name: Churn Score
      dtype: int32
    - name: CLTV
      dtype: float32
    - name: Churn Category
      dtype: string
    - name: Churn Reason
      dtype: string
    - name: Partner
      dtype: bool
  config_name: default
  splits:
    - name: train
      num_bytes: 121430 # Replace with actual value
      num_examples: 7043
  download_size: 121430 # Replace with actual value
  dataset_size: 121430 # Replace with actual value
train-eval-index:
  - config: default
    task: tabular-classification
    task_id: multi_class_classification
    splits:
      train_split: train
      eval_split: validation # Update if you have a validation split
    col_mapping:
      label: Churn Label
    metrics:
      - type: accuracy
        name: Accuracy

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
