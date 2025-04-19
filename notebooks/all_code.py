#%% [markdown]
## Improrts and Data Information #### 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import spearmanr

# for machine learning 
# !pip install pycaret
from pycaret.classification import *
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, precision_score, f1_score, recall_score, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report

#%% [markdown]
## Feature Information
# ID: ID of each client
#
# LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
#
# SEX: Gender (1=male, 2=female)
# 
# EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
# 
# MARRIAGE: Marital status (1=married, 2=single, 3=others)
# 
# AGE: Age in years
# 
# PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)
# 
# PAY_2: Repayment status in August, 2005 (scale same as above)
# 
# PAY_3: Repayment status in July, 2005 (scale same as above)
# 
# PAY_4: Repayment status in June, 2005 (scale same as above)
# 
# PAY_5: Repayment status in May, 2005 (scale same as above)
# 
# PAY_6: Repayment status in April, 2005 (scale same as above)
# 
# BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
# 
# BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
# 
# BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
# 
# BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
# 
# BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
# 
# BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
# 
# PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
# 
# PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
# 
# PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
# 
# PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
# 
# PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
# 
# PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
# 
# default.payment.next.month: Default payment (1=yes, 0=no)
# 
## Data Preprocessing
#%%
# Functions
def load_data(file_path):
    """Load data from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def replace_col_name(df, col_name_mapping):
    df.rename(columns=col_name_mapping, inplace=True)
    return df

def create_risk_binary(row, limit_mean, last_3_months=["Sept_Pay_status", "August_Pay_status", "July_Pay_status"]):
    high_limit = row["LIMIT_BAL"] > limit_mean
    
    # Check if payments were made in the last 3 months
    # Assuming no delay in payment status indicates payments were made on time
    # last_3_months = ["Sept_Pay_status", "August_Pay_status", "July_Pay_status"]
    good_payment = all(row[col] <= 0 for col in last_3_months)  # No delays
    
    if high_limit and good_payment:
        return 1
    else:
        return 0

# Function to calculate momentum by row
def calculate_weighted_momentum(row, pay_status_cols=pay_status_cols):
    weights = [1, 2, 3, 4, 5, 6]  # Weights for April to September, increasing the closer to October
    weighted_sum = sum(weights[i] * row[pay_status_cols[i]] for i in range(len(pay_status_cols)))
    # return 1 if weighted_sum > 6 else 0  # Adjust threshold as needed
    # return 1 if weighted_sum > 10 else 0  # Adjust threshold as needed
    return 1 if weighted_sum > 20 else 0  # Adjust threshold as needed

# Function to plot correlation
def plot_correlation(df, feature_cols, method="pearson", target_col = "default_payment_next_month"):
    columns = feature_cols + [target_col]

    # Compute correlation matrix
    correlation_matrix = df[columns].corr(method=method)

    target_correlation = correlation_matrix[target_col].sort_values(ascending=False)

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Matrix with Target: default_payment_next_month")
    plt.show()

    return target_correlation

#%%
# Load our data 
clients_data = load_data("hf://datasets/scikit-learn/credit-card-clients/UCI_Credit_Card.csv")

#%% 
# Clean Data

# Drop rows with missing data
clients_data.dropna(inplace=True)

column_mappings = {
    # "default.payment.next.month": "default",
    "default.payment.next.month": "default_payment_next_month",
    "PAY_0": "Sept_Pay_status",
    "PAY_2": "August_Pay_status",
    "PAY_3": "July_Pay_status",
    "PAY_4": "June_Pay_status",
    "PAY_5": "May_Pay_status",
    "PAY_6": "April_Pay_status",
    "BILL_AMT1": "Sept_Bill_Amount",
    "BILL_AMT2": "August_Bill_Amount",
    "BILL_AMT3": "July_Bill_Amount",
    "BILL_AMT4": "June_Bill_Amount",
    "BILL_AMT5": "May_Bill_Amount",
    "BILL_AMT6": "April_Bill_Amount",
    "PAY_AMT1": "Sept_Pay_Amount",
    "PAY_AMT2": "August_Pay_Amount",
    "PAY_AMT3": "July_Pay_Amount",
    "PAY_AMT4": "June_Pay_Amount",
    "PAY_AMT5": "May_Pay_Amount",
    "PAY_AMT6": "April_Pay_Amount"
}

# Rename columns to easier to understand names
clients_data = replace_col_name(clients_data, column_mappings)

# Convert NT dollars to USD (1 USD = 31.8 NTD)
conversion_rate = 1 / 31.8 

# List of columns to convert
nt_dollar_columns = [
    "LIMIT_BAL", "Sept_Bill_Amount", "August_Bill_Amount", "July_Bill_Amount",
    "June_Bill_Amount", "May_Bill_Amount", "April_Bill_Amount",
    "Sept_Pay_Amount", "August_Pay_Amount", "July_Pay_Amount",
    "June_Pay_Amount", "May_Pay_Amount", "April_Pay_Amount"
]

# Convert NT dollar columns to US dollars
for col in nt_dollar_columns:
    clients_data[col] = clients_data[col] * conversion_rate

# Verify the conversion
clients_data[nt_dollar_columns].head()

# Calculate momentum based on payment status
# Define the payment status columns in chronological order
pay_status_cols = [
    "April_Pay_status", "May_Pay_status", "June_Pay_status",
    "July_Pay_status", "August_Pay_status", "Sept_Pay_status"
]



clients_data["Momentum"] = clients_data.apply(calculate_weighted_momentum, axis=1)
clients_data["Momentum_Label"] = clients_data["Momentum"].map({0: "Stable/Improving", 1: "Bad Momentum"})

# View the resulting dataframe
clients_data.head()

#%%
# Let's remove columns that we don't need 
# We don't need ID, as it's just a unique identifier
# We also don't need SEX, as we can't discriminate by SEX, but we could do it later in Machine Learning part. 
clients_data.drop("ID", inplace = True, axis = 1)

#%%
# Let's Check Marriage variable
clients_data["MARRIAGE"].value_counts(
    # normalize = True gives numbers in proportions, which is very useful for the analysis
    normalize= True )

# 2 - Single - ~0.54% of the data
# 1 - Married - ~0.46% of the data 
# 3/0 - anything else is other and very arbitrary les than 2% of the data, we should drop such instancess

# drop 3 and 0 class in marriage variable
clients_data = clients_data[~clients_data["MARRIAGE"].isin([0, 3])]
#%% Check the change of Marriage Variable value counts 
# check
clients_data["MARRIAGE"].value_counts(normalize= True)

# It doesn't have an ordinal relationship as it is a nominal categorical variable.
# Let's convert that to binary
# That's fine now!

# Marriage doesn't have an ordinal relationship as it is a pure nominal categorical variable.
# Let's convert that to binary
# 1 - Married
# 2 - Single
clients_data["MARRIAGE"] = clients_data["MARRIAGE"].astype("category")

# %% 
# Education Variable 
# 1 - graduate
# 2 - undergraduate
# 3 - high school
# anything else - other/arbitrary 

# Check unique values as well
clients_data["EDUCATION"].value_counts(normalize= True)

# Less than 1% of the data are - 4, 5, 6, 0 which are not explained and arbitrary. 
# We should drop such instances as well. 
#%%
# Leave only required instances of education
clients_data = clients_data[clients_data["EDUCATION"].isin([1, 2, 3])]
# Check unique values again
clients_data["EDUCATION"].value_counts(normalize= True)

#%%[markdown]
## SMART Questions
# 1. How does the payment status trend affect the likelihood of defaulting in October 2005?
# 
# 2. Does limit balance affect the likelihood of defaulting in October 2005?
# 
# 3. Does having a higher credit limit reduce the likelihood of a client defaulting in October 2005, based on financial behavior from the previous six months?
#
# 4. Do age, sex, marriage, or education levels have any impact on the likelihood of defaulting in October 2005?
# 

## Exploratory Data Analysis (EDA)
#%%
# Credit Limit Distribution
# pretty right-skewed distribution, might need to be transformed with log later. 
plt.figure(figsize= (12, 10))

sns.histplot(data = clients_data, 
             x = "LIMIT_BAL")

# design 
plt.title("Overall distribution of Credit Limit", 
          fontdict= {"fontsize":30, 
                     "fontweight":"bold"})
plt.xlabel("Credit Limit", 
           fontdict= {"fontsize": 20})

plt.ylabel("Frequency", 
           fontdict= {"fontsize": 20})

plt.tick_params(axis='both', labelsize=15)

plt.show()

plt.figure(figsize= (12, 10))

# Credit Limit vs Default Status
# Overall, Defaulting clients tend to have on average less than 5000 USD credit limit
# Non-defaulted cliennts have a median credit limit of at least 5000 uSD. 
sns.boxplot(data = clients_data, 
            y = "default_payment_next_month",
             x = "LIMIT_BAL", 
             orient= "h")

# design 
plt.title("Overall distribution of Credit Limit by Default Status", 
          fontdict= {"fontsize":30, 
                     "fontweight":"bold"})
plt.xlabel("Credit Limit", 
           fontdict= {"fontsize": 20})

plt.ylabel("Default Status", 
           fontdict= {"fontsize": 20})

plt.tick_params(axis='both', labelsize=15)

plt.show()

# Credit Limit vs Default Status by Sex and Marriage status
plt.figure(figsize= (12, 10))

plt.subplot(1, 2, 1)

sns.boxplot(data = clients_data, 
            y = "default_payment_next_month",
             x = "LIMIT_BAL", 
             orient= "h", 
             hue = "SEX")

# design 
plt.xlabel("Credit Limit", 
           fontdict= {"fontsize": 20})

plt.ylabel("Default Status", 
           fontdict= {"fontsize": 20})

plt.tick_params(axis='both', labelsize=15)

plt.subplot(1, 2, 2)
sns.boxplot(data = clients_data, 
            y = "default_payment_next_month",
             x = "LIMIT_BAL", 
             orient= "h", 
             hue = "MARRIAGE")

# design 
plt.xlabel("Credit Limit", 
           fontdict= {"fontsize": 20})

plt.ylabel("Default Status", 
           fontdict= {"fontsize": 20})

plt.tick_params(axis='both', labelsize=15)

plt.show()

#%%
# Correlation matrix of Age, sex, marriage, and education with defaulting
client_background_correlation_result = plot_correlation(clients_data, ["AGE", "SEX", "MARRIAGE", "EDUCATION"], method="spearman")
print(client_background_correlation_result)

## Background Correlation
# There are no significant correlations to defaulting in October 2005 from the background data.

#%%
# Limit balance correlation
client_limit_balance_correlation_result = plot_correlation(clients_data, ["LIMIT_BAL"], method="spearman")
print(client_limit_balance_correlation_result)
