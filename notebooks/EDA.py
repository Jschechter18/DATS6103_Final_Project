#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


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


# %%
# Correlation matrix
correlation_cols = ["AGE", "LIMIT_BAL", "SEX", "MARRIAGE", "EDUCATION", "default_payment_next_month"]

correlation_matrix = clients_data[correlation_cols].corr()
target_correlation = correlation_matrix["default_payment_next_month"].sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix")
plt.show()

# %%
# Distributions

# Age
plt.figure()
sns.histplot(clients_data["AGE"], bins=50, kde=True)
plt.title("Age Distribution")


#%%
# Amount of bill statement in September, 2005
plt.figure()
sns.histplot(clients_data["Sept_Bill_Amount"], bins=50, kde=True)
plt.title("Bill Amount in September, 2005 Distribution")
plt.xlim(0, 20000)

# Amount of bill statement in August, 2005
plt.figure()
sns.histplot(clients_data["August_Bill_Amount"], bins=50, kde=True)
plt.title("Bill Amount in August, 2005 Distribution")
plt.xlim(0, 20000)

# Amount of bill statement in July, 2005
plt.figure()
sns.histplot(clients_data["July_Bill_Amount"], bins=50, kde=True)
plt.title("Bill Amount in July, 2005 Distribution")
plt.xlim(0, 20000)

# Amount of bill statement in June, 2005
plt.figure()
sns.histplot(clients_data["June_Bill_Amount"], bins=50, kde=True)
plt.title("Bill Amount in June, 2005 Distribution")
plt.xlim(0, 20000)

# Amount of bill statement in May, 2005
plt.figure()
sns.histplot(clients_data["May_Bill_Amount"], bins=50, kde=True)
plt.title("Bill Amount in May, 2005 Distribution")
plt.xlim(0, 20000)

# Amount of bill statement in April, 2005
plt.figure()
sns.histplot(clients_data["April_Bill_Amount"], bins=50, kde=True)
plt.title("Bill Amount in April, 2005 Distribution")
plt.xlim(0, 20000)

# %%
