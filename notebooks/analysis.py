#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import spearmanr


#%%[markdown]
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

# Function to calculate momentum by row
def calculate_weighted_momentum(row):
    weights = [1, 2, 3, 4, 5, 6]  # Weights for April to September, increasing the closer to October
    weighted_sum = sum(weights[i] * row[pay_status_cols[i]] for i in range(len(pay_status_cols)))
    # return "Bad Momentum" if weighted_sum > 20 else "Stable/Improving"  # Adjust threshold as needed
    # 0 is good, 1 is bad
    return 1 if weighted_sum > 20 else 0  # Adjust threshold as needed

#%%
# Load data
clients_data = load_data("../data/credit_card_clients_full.csv")

# Drop unnecessary columns
clients_data.drop(columns=["Unnamed: 0"], inplace=True, axis=1)

# %%
# Clean data

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

# Specify categorical columns
categorical_columns = ["SEX", "EDUCATION", "MARRIAGE", "Sept_Pay_status",
                       "August_Pay_status", "July_Pay_status", "June_Pay_status",
                       "May_Pay_status", "April_Pay_status", "default_payment_next_month"]

for col in categorical_columns:
    clients_data[col] = clients_data[col].astype("category")



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

# Correlation matrix of Age, sex, marriage, and education with defaulting
client_background_correlation_result = plot_correlation(clients_data, ["AGE", "SEX", "MARRIAGE", "EDUCATION"], method="spearman")
print(client_background_correlation_result)

#%%[markdown]
## Background Correlation
# There are no significant correlations to defaulting in October 2005 from the background data.

#%%
# Limit balance correlation
client_limit_balance_correlation_result = plot_correlation(clients_data, ["LIMIT_BAL"], method="spearman")
print(client_limit_balance_correlation_result)

#%%[markdown]
## Limit Balance Correlation
# The limit balance has a moderate weak correlation with defaulting in October 2005.

#%%
# Creating an average of Pay_status(Pay_)
pay_status_cols = [
    "Sept_Pay_status",
    "August_Pay_status",
    "July_Pay_status",
    "June_Pay_status",
    "May_Pay_status",
    "April_Pay_status"
]

# Creating an average of Pay_anount
pay_amount_cols = [
    "Sept_Pay_Amount",
    "August_Pay_Amount",
    "July_Pay_Amount",
    "June_Pay_Amount",
    "May_Pay_Amount",
    "April_Pay_Amount"
]

# Creating an average of Bill_amount
bill_amount_cols = [
    "Sept_Bill_Amount",
    "August_Bill_Amount",
    "July_Bill_Amount",
    "June_Bill_Amount",
    "May_Bill_Amount",
    "April_Bill_Amount"
]

clients_data["Mean_Pay_Amount"] = clients_data[pay_amount_cols].mean(axis=1)
clients_data["Mean_Bill_Amount"] = clients_data[bill_amount_cols].mean(axis=1)

# correlation_result = plot_correlation(clients_data, ["AGE", "LIMIT_BAL", "SEX", "MARRIAGE", "EDUCATION", "Mean_Pay_Status", "Mean_Pay_Amount", "Mean_Bill_Amount"])
client_spending_correlation_result = plot_correlation(clients_data, ["Mean_Pay_Amount", "Mean_Bill_Amount"])
print(client_spending_correlation_result)

#%%[markdown]
## Spending Correlation
# The mean pay amount has a weak negative correlation with defaulting in October 2005.
# Mean bill amount has no relationship with defaulting in October 2005.

#%%
client_monthly_pay_status_correlation_result = plot_correlation(clients_data, ["Sept_Pay_status","August_Pay_status","July_Pay_status", "June_Pay_status", "May_Pay_status","April_Pay_status"], method="spearman")
print(client_monthly_pay_status_correlation_result)

#%%[markdown]
## Payment Status Correlation
# The payment status has a modertate positive correlation with defaulting in October 2005 for all of the months.
# It becomes stronger in the months closer to October

#%%
# Momentum Correlation
client_momentum_correlation_result = plot_correlation(clients_data, ["Momentum"], method="spearman")
print(client_momentum_correlation_result)

#%%[markdown]
## Momentum Correlation
# The momentum has a moderate positive correlation with defaulting in October 2005.
# It is the best predictor we have based on correlation.

#%%[markdown]
## Observations thus far:
# 1. The most important feature for predicting default is momentum
# 2. The next is the monthly pay status, especially for months approaching October. Note, momentum accounts for this with weights.
# 3. The final relevant feature is the limit balance.
# 4. The background data has no significant correlation with defaulting in October 2005.


#%%
# Plotting
plt.figure(figsize=(8, 6))
sns.histplot(clients_data["LIMIT_BAL"], kde=True, bins=30, color="blue")
plt.title("Distribution of Credit Limit (LIMIT_BAL)")
plt.xlabel("Credit Limit (USD)")
plt.ylabel("Frequency")
plt.show()

# Plotting the distribution of default payments to April
plt.figure(figsize=(8, 6))
sns.histplot(data=clients_data, x="April_Pay_status", hue="default_payment_next_month", multiple="stack", bins=30, palette="Set2")
plt.title("Distribution of Default Payments in April by Payment Status")
plt.xlabel("April Payment Status")
plt.ylabel("Frequency")
plt.show()

# Plotting the distribution of default payments to September
plt.figure(figsize=(8, 6))
sns.histplot(data=clients_data, x="Sept_Pay_status", hue="default_payment_next_month", multiple="stack", bins=30, palette="Set2")
plt.title("Distribution of Default Payments in September by Payment Status")
plt.xlabel("September Payment Status")
plt.ylabel("Frequency")
plt.show()

# Display count of default payments
default_counts = clients_data["default_payment_next_month"].value_counts()
print(f"Default Payment Counts: {default_counts}")

#%%
momentum_proportions =  (
    clients_data.groupby("Momentum_Label")["default_payment_next_month"]
    .value_counts(normalize=True)
    .rename("proportion")
    .reset_index()
)
plt.figure(figsize=(8, 6))
sns.barplot(data=momentum_proportions,x="Momentum_Label", y="proportion", hue="default_payment_next_month")
plt.title("Distribution of October Default Payments by Momentum")
plt.xlabel("Momentum")
plt.ylabel("Frequency")
plt.ylim(0.0, 1.0)
plt.legend(title="Payment Defaulted in October")
plt.show()

# Show the proportion of default payments by momentum
print(clients_data.groupby("Momentum")["default_payment_next_month"].value_counts(normalize=True))
#%%[markdown]
# Bad Momentum  
#    
# 0: 0.358769  -  1: 0.641231
# 
# Stable/Improving
# 
# 0: 0.829832  -  1: 0.170168
# 
# There is a slight uptick in likelihood of defaulting when the momentum is bad.

# %%[markdown]
## EDA Summary
# Clients who defaulted in October, were more likely to have later payment status in months closer to October.
# This indicates that payment status is a good predictor of default.
#
# There is a relationship between momentum and default payment.
# When momentum is bad, we hypothesize that the client is more likely to default.
# 
# There is a relationship between high limit balance and default payment.
# As limit balance increases, the odds of defaulting also increase.

# %%
