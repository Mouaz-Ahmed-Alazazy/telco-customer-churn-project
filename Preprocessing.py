#top highest corelation features with target variable 'Churn'
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

corr_matrix = df.corr(numeric_only=True)

# identifying top features correlated with 'Churn'
corr_target = corr_matrix['Churn'].abs()
top_features = corr_target[corr_target >= 0.30].index.tolist()
df_selected_test = df_test[top_features]
print("Top features correlated with 'Churn':", top_features)

# selecting only the top features from the original dataframe
df_selected = df[top_features]
print(df_selected.head())

# check data imblance in target variable
churn_counts = df['Churn'].value_counts()
print(churn_counts)

# Data Preprocessing and SMOTE
scaler = StandardScaler()
x = df_selected.drop('Churn', axis=1)
y = df_selected['Churn']

# Splitting the data into training and testing sets
train = df_selected.copy()
x_train = train.drop('Churn', axis=1)
y_train = train['Churn']

x_test = df_selected_test.drop('Churn', axis=1)
y_test = df_selected_test['Churn']

x_train_scaled = scaler.fit_transform(x_train)
X_test_scaled  = scaler.transform(x_test)

# Applying SMOTE to balance the classes
sm = SMOTE(random_state=42)

X_train_res, y_train_res = sm.fit_resample(x_train_scaled, y_train)
print("After SMOTE, counts of label '1': {}".format(sum(y_train_res==1)))
print("After SMOTE, counts of label '0': {} \n".format(sum(y_train_res==0)))