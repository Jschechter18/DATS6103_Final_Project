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

# Function to calculate momentum
def calculate_momentum(row):
    # Check if payment status is worsening month-to-month
    worsening = all(row[pay_status_cols[i]] <= row[pay_status_cols[i + 1]] for i in range(len(pay_status_cols) - 1))
    return "Bad Momentum" if worsening else "Stable/Improving"

# Apply the function to create the momentum column
clients_data["Momentum"] = clients_data.apply(calculate_momentum, axis=1)
# clients_data["Momentum"] = clients_data.apply(calculate_momentum, axis=1)

# View the resulting dataframe
clients_data.head()



#%%[markdown]
## SMART Questions
# 1. What features are the best predictors of defaulting payments?
#
# 2. How does level of education and marriage status correlate with the clients risk of defaulting on their payments?
#
# 3. Are younger clients more likely to default on their payments than older clients?
#
# 4. Does having a higher credit limit reduce the likelihood of a client defaulting in October 2005, based on financial behavior from the previous six months?
#
# 5. Do clients who make the minimum payments month have a higher likelihood of defaulting at any given month?
#
## Exploratory Data Analysis (EDA)

#%%

# Correlation matrix
correlation_result = plot_correlation(clients_data, ["AGE", "SEX", "MARRIAGE", "EDUCATION"], method="spearman")
print(correlation_result)

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

# clients_data["Mean_Pay_Status"] = clients_data[pay_status_cols].mean(axis=1)

# Creating an average of Pay_anount
pay_amount_cols = [
    "Sept_Pay_Amount",
    "August_Pay_Amount",
    "July_Pay_Amount",
    "June_Pay_Amount",
    "May_Pay_Amount",
    "April_Pay_Amount"
]

clients_data["Mean_Pay_Amount"] = clients_data[pay_amount_cols].mean(axis=1)

# Creating an average of Bill_amount
bill_amount_cols = [
    "Sept_Bill_Amount",
    "August_Bill_Amount",
    "July_Bill_Amount",
    "June_Bill_Amount",
    "May_Bill_Amount",
    "April_Bill_Amount"
]

clients_data["Mean_Bill_Amount"] = clients_data[bill_amount_cols].mean(axis=1)

# correlation_result = plot_correlation(clients_data, ["AGE", "LIMIT_BAL", "SEX", "MARRIAGE", "EDUCATION", "Mean_Pay_Status", "Mean_Pay_Amount", "Mean_Bill_Amount"])
correlation_result = plot_correlation(clients_data, ["AGE", "LIMIT_BAL", "SEX", "MARRIAGE", "EDUCATION", "Mean_Pay_Amount", "Mean_Bill_Amount"])
print(correlation_result)

# Conclusion: Mean Pay status has a strong correlation with the target column


#%%
correlation_result = plot_correlation(clients_data, ["Sept_Pay_status","August_Pay_status","July_Pay_status", "June_Pay_status", "May_Pay_status","April_Pay_status"], method="spearman")
print(correlation_result)

# %%[markdown]
# EDA Takeaways so far:

# 1. The most important features for predicting default are the average payment status in the last 3 months
# From here we can consider momentum
# 2. The limit balance is a good predictor of default.
# We found no other useful features that are not already in the dataset


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
# I want to plot odds of defaulting based on momentum
plt.figure(figsize=(8, 6))
sns.histplot(data=clients_data, x="Momentum", hue="default_payment_next_month", multiple="stack", bins=30, palette="Set2")
plt.title("Distribution of Default Payments by Momentum")
plt.xlabel("Momentum")
plt.ylabel("Frequency")
plt.show()

# Show the proportion of default payments by momentum
print(clients_data.groupby("Momentum")["default_payment_next_month"].value_counts(normalize=True))
#%%[markdown]
# Bad Momentum  
#    
# 0: 0.808059
# 
# 1: 0.191941
# 
# Stable/Improving
# 
# 0: 0.726717
# 
# 1: 0.273283
# 
# There is a slight uptick in likelihood of defaulting when the momentum is bad.

# %%[markdown]
# Clients who defaulted in October, were more likely to have later payment status in September.
# This reinforces the idea that payment status is a good predictor of default.
#
# There is a weak, but present correlation between momentum and default payment.

