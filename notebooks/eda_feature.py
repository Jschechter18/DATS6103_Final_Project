#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy import stats


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



#%%
# Functions
def load_data(file_path):
    """Load data from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def replace_col_name(df, col_name_mapping):
    df.rename(columns=col_name_mapping, inplace=True)
    return df


# Create the binary feature (1 = lower risk, 0 = higher risk)
def create_risk_binary(row, limit_mean):
    high_limit = row["LIMIT_BAL"] > limit_mean
    
    # Check if payments were made in the last 3 months
    # Assuming no delay in payment status indicates payments were made on time
    last_3_months = ["Sept_Pay_status", "August_Pay_status", "July_Pay_status"]
    good_payment = all(row[col] <= 0 for col in last_3_months)  # No delays
    
    if high_limit and good_payment:
        return 1
    else:
        return 0
    
def plot_correlation(feature_cols):
    target_col = "default_payment_next_month"
    columns = feature_cols + [target_col]

    # Compute correlation matrix
    correlation_matrix = clients_data[columns].corr()

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

#%%[markdown]
## Data Preprocessing

# %%
# Clean data

# Drop rows with missing data
clients_data.dropna(inplace=True)

column_mappings = {
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


# View the resulting dataframe
clients_data.head()

#%%

'''
Reducing the floating point to two places
'''
# clients_data.info()

float_cols = clients_data.select_dtypes(include=['float64']).columns
for col in float_cols:
    clients_data[col] = clients_data[col].round(2)

# clients_data.head()

#%%
# Functions:


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

'''
Correlation matrix
'''

#%%



# %%
# Correlation matrix
correlation_result = plot_correlation(["AGE", "LIMIT_BAL", "SEX", "MARRIAGE", "EDUCATION"])
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

clients_data["Mean_Pay_Status"] = clients_data[pay_status_cols].mean(axis=1)

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

# clients_data.head()

correlation_result_update_one = plot_correlation(["AGE", "LIMIT_BAL", "SEX", "MARRIAGE", "EDUCATION", "Mean_Pay_Status", "Mean_Pay_Amount", "Mean_Bill_Amount"])
print(correlation_result_update_one)

# Conclusion: Mean Pay status has a strong correlation with the target column

#%%
correlation_result_update_two = plot_correlation(["Sept_Pay_status","August_Pay_status","July_Pay_status", "June_Pay_status", "May_Pay_status","April_Pay_status"])
print(correlation_result_update_two)

# Conclusion: Clear pattern where more recent payment statuses have stronger correlations with default

#%%

# Creating an average of Pay_status(Pay_) for April, May, June
pay_status_cols_first_half = [
    "June_Pay_status",
    "May_Pay_status",
    "April_Pay_status"
]

clients_data["Mean_Pay_Status_First_Half"] = clients_data[pay_status_cols_first_half].mean(axis=1)

# Convert the categorical columns to numeric first, then calculate mean
pay_status_cols_second_half = [
    "Sept_Pay_status",
    "August_Pay_status",
    "July_Pay_status",
]

# Create numeric versions of the columns
for col in pay_status_cols_second_half:
    # Create a new numeric column based on the categorical one
    clients_data[col + "_numeric"] = clients_data[col].astype(int)

# Use the numeric versions to calculate the mean
numeric_cols_second_half = [col + "_numeric" for col in pay_status_cols_second_half]
clients_data["Mean_Pay_Status_Second_Half"] = clients_data[numeric_cols_second_half].mean(axis=1)


correlation_result_update_three = plot_correlation(["Mean_Pay_Status_First_Half", "Mean_Pay_Status_Second_Half", "LIMIT_BAL"])
print(correlation_result_update_three)

# Conclusion: Recent payment status history should be given more weight
# Conclusion: Credit limit provides valuable additional predictive power

#%%
correlation_result_update_four = plot_correlation(["AGE", "Mean_Pay_Status_First_Half", "Mean_Pay_Status_Second_Half", "LIMIT_BAL"])
print(correlation_result_update_four)

# Check the default column for mean_second_half + Limit_bal

#%%
##########
##########
# Spearman correlation matrix
#########
#########

# Columns to analyze
payment_cols = [
    "Sept_Pay_status", 
    "August_Pay_status", 
    "July_Pay_status",
    "Mean_Pay_Status_Second_Half",
    "Mean_Pay_Status_First_Half",
    "LIMIT_BAL",
    "default_payment_next_month"
]

# Spearman correlation matrix
spearman_corr = clients_data[payment_cols].corr(method='spearman')

plt.figure(figsize=(10, 8))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Spearman Rank Correlation Matrix')
plt.tight_layout()
plt.show()

default_correlations = spearman_corr["default_payment_next_month"].drop("default_payment_next_month").sort_values(ascending=False)
print("Spearman correlations with default payment (strongest to weakest):")
print(default_correlations)


# The correlation confirms that the recent payment behavior is the strongest predictor of default risk


#%%

continuous_vars = [
    "LIMIT_BAL", 
    "Mean_Pay_Status_First_Half", 
    "Mean_Pay_Status_Second_Half",
    "Sept_Pay_status",  
    "August_Pay_status",  
    "July_Pay_status"
]

pointbiserial_results = []

# Calculate Point-Biserial correlation for each variable with default
for var in continuous_vars:
    correlation, pvalue = stats.pointbiserialr(
        clients_data["default_payment_next_month"], 
        clients_data[var]
    )
    pointbiserial_results.append({
        'Variable': var,
        'Correlation': correlation,
        'p-value': pvalue
    })

results_df = pd.DataFrame(pointbiserial_results)
results_df = results_df.sort_values('Correlation', ascending=False)

print("Point-Biserial Correlation with Default Payment:")
print(results_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='Correlation', y='Variable', data=results_df)
plt.title('Point-Biserial Correlation with Default Payment')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.show()

#%%

clients_data.describe()

limit_mean = clients_data["LIMIT_BAL"].mean()
print(f"Mean LIMIT_BAL: {limit_mean}")



# Apply the function to create the new binary feature
clients_data["Low_Risk_Flag"] = clients_data.apply(create_risk_binary, axis=1)

correlation_result_update_five = plot_correlation(["Low_Risk_Flag"])
print(correlation_result_update_five)


# clients_data["Mean_Bill_Amount"]
# clients_data['LIMIT_BAL']
#%%

clients_data['threshhold_amt'] = clients_data['LIMIT_BAL']*.9

def update_col(row):
    # row[col] = row[limit bal] - row[mean bal] < row[threshold]
    return row['Mean_Bill_Amount'] >= row['threshhold_amt'] 
    # print(t)
        
clients_data["Approached_Limit"] =  clients_data.apply(update_col, axis=1) 


correlation_result_update_three = plot_correlation(["Mean_Pay_Status_Second_Half", "Approached_Limit"])
print(correlation_result_update_three)

clients_data.info()


# high Lmit_bal vs Low education value

#%%
"""
- high Lmit_bal vs Low education value
- if the customer is single/divorced and has a higher bill amount 
with bad momentum there is a chance he might default
"""


#%%
# Define high limit balance threshold (e.g., above the mean)
limit_mean = clients_data["LIMIT_BAL"].mean()
clients_data["High_Limit_Balance"] = clients_data["LIMIT_BAL"] > limit_mean

# Filter for low education levels (e.g., high school or others)
clients_data["Low_Education"] = clients_data["EDUCATION"].isin([3, 4])

# Crosstab to analyze the relationship
crosstab = pd.crosstab(
    [clients_data["High_Limit_Balance"], clients_data["Low_Education"]],
    clients_data["default_payment_next_month"],
    normalize="index"
)

print("Crosstab of High Limit Balance, Low Education, and Default Payment:")
print(crosstab)
# %%[markdown]
## EDA Conclusion
# 
# 1.
#%%
