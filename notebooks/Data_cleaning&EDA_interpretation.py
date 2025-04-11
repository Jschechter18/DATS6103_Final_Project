#%%
from datasets import load_dataset
import pandas as pd

# Load the dataset
dataset = load_dataset("scikit-learn/credit-card-clients", split='train')

# Convert to a pandas DataFrame
df = dataset.to_pandas()

# Save the dataset to CSV
df.to_csv("credit_card_clients_full.csv", index=False)
#%%

# Display basic dataset info
df.head(), df.info() 


# Check for missing values
print("\n Missing values per column:")
print(df.isnull().sum())
#%%


#%%
# We will:

# Rename columns clearly.

# Check and handle missing values.

# Verify data consistency (correct categorical encodings, outliers)

# Let's start with renaming columns for clarity

# Renaming columns clearly and consistently


df.rename(columns={
    'ID': 'Client_ID',
    'LIMIT_BAL': 'Credit_Limit',
    'SEX': 'Gender',
    'EDUCATION': 'Education_Level',
    'MARRIAGE': 'Marital_Status',
    'AGE': 'Age',
    'PAY_0': 'Repayment_Status_Sep',
    'PAY_2': 'Repayment_Status_Aug',
    'PAY_3': 'Repayment_Status_Jul',
    'PAY_4': 'Repayment_Status_Jun',
    'PAY_5': 'Repayment_Status_May',
    'PAY_6': 'Repayment_Status_Apr',
    'BILL_AMT1': 'Bill_Amount_Sep',
    'BILL_AMT2': 'Bill_Amount_Aug',
    'BILL_AMT3': 'Bill_Amount_Jul',
    'BILL_AMT4': 'Bill_Amount_Jun',
    'BILL_AMT5': 'Bill_Amount_May',
    'BILL_AMT6': 'Bill_Amount_Apr',
    'PAY_AMT1': 'Payment_Amount_Sep',
    'PAY_AMT2': 'Payment_Amount_Aug',
    'PAY_AMT3': 'Payment_Amount_Jul',
    'PAY_AMT4': 'Payment_Amount_Jun',
    'PAY_AMT5': 'Payment_Amount_May',
    'PAY_AMT6': 'Payment_Amount_Apr',
    'default.payment.next.month': 'Default_Payment'
}, inplace=True)


# Verify the changes
print(df.columns)

# Let Check for missing values clearly
missing_values = df.isnull().sum()
print("\nMissing values per column:\n", missing_values)

# Let Check unique values in categorical columns using the updated names
categorical_cols = ['Gender', 'Education_Level', 'Marital_Status', 'Default_Payment']
unique_values = {col: df[col].unique() for col in categorical_cols}

print("\nUnique values in categorical columns:\n", unique_values)
# %%

# Observations from Data Cleaning:
# Missing Values:
# No missing values detected (great news).


# Categorical Variables:
# SEX: Encoded as [1, 2] representing Male and Female.
# EDUCATION: Encoded as [1, 2, 3, 4, 5, 6, 0]. There seems to be some unexpected encoding (0, 5, 6 may be unclear or invalid categories).
# MARRIAGE: Encoded as [0, 1, 2, 3]. Category 0 is not standard and may represent missing or unknown.


# Action for unclear categories:
# For EDUCATION:
# Categories: 1 = graduate school, 2 = university, 3 = high school, 4 = others
# Categories 0, 5, 6 can be consolidated into category 4 (others).

# For MARRIAGE:
# Categories: 1 = married, 2 = single, 3 = others, Category 0 can be grouped with 3 (others).

# Let's perform these corrections: ​

#%%
df['Education_Level'] = df['Education_Level'].replace({0:4, 5:4, 6:4})
df['Marital_Status'] = df['Marital_Status'].replace({0:3})
education_counts = df['Education_Level'].value_counts()
marriage_counts = df['Marital_Status'].value_counts()

education_counts, marriage_counts
# %%

# Categorical Corrections Completed:

# EDUCATION: 
# Graduate school (1): 10,585
# University (2): 14,030
# High school (3): 4,917
# Others (4): 468 (includes previous unclear values)

# MARRIAGE:
# Married (1): 13,659
# Single (2): 15,964
# Others (3): 377 (including the previously unclear category 0)


# Exploratory Data Analysis (EDA):

# Next, let's visualize some key aspects to understand patterns in our data clearly:

# Distribution of default payments
# Distribution of age
# Distribution by education and default payments
# Distribution by marital status and default payments
# Credit limit distribution by default status

#%% 
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")

# Creating subplots clearly for EDA visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plotting Default Payment Distribution
sns.countplot(x='Default_Payment', data=df, ax=axes[0,0])
axes[0,0].set_title('Default Payment Distribution')
axes[0,0].set_xticklabels(['No Default', 'Default'])

# Age Distribution
sns.histplot(df['Age'], bins=30, kde=True, ax=axes[0,1])
axes[0,1].set_title('Age Distribution')

# Education vs Default Payment
sns.countplot(x='Education_Level', hue='Default_Payment', data=df, ax=axes[1,0])
axes[1,0].set_title('Education Level vs Default Payment')
axes[1,0].set_xticklabels(['Graduate School', 'University', 'High School', 'Others'])

# Marital Status vs Default Payment
sns.countplot(x='Marital_Status', hue='Default_Payment', data=df, ax=axes[1,1])
axes[1,1].set_title('Marital Status vs Default Payment')
axes[1,1].set_xticklabels(['Married', 'Single', 'Others'])

# Adjust layout
plt.tight_layout()
plt.show()

# Credit Limit distribution by Default Payment (separate plot for clarity)
plt.figure(figsize=(8,6))
sns.boxplot(x='Default_Payment', y='Credit_Limit', data=df)
plt.title('Credit Limit Distribution by Default Status')
plt.xticks([0, 1], ['No Default', 'Default'])
plt.show()
# %%


#%%
# Interpretation for each graph:

# Default Payment Distribution:
# The graph shows the number of clients who defaulted versus those who did not default on their credit card payment in the next month.
# The majority of clients did not default (label = 0), indicating the dataset is imbalanced.
# A smaller portion of clients defaulted (label = 1).
# This imbalance is important to consider during model training and evaluation, as it may bias predictive models toward the majority class.


# Age Distribution:
# The graph illustrates the distribution of client ages.
# Most clients fall within the 20 to 40 years age range, with a peak around the early 30s.
# The distribution is right-skewed, with fewer clients aged above 60.
# Understanding the age distribution is useful for identifying which age groups are most represented and potentially more at risk of defaulting.


# Education Level vs. Default Payment:
# The chart compares default behavior across different education levels:
# Clients with graduate school and university education have higher counts overall.
# However, the proportion of defaults within each education group suggests that clients with high school or 'other' education may have a slightly higher risk of defaulting.
# This may suggest a correlation between lower education levels and higher credit risk, which can be statistically confirmed with a chi-square test.


# Marital Status vs. Default Payment:
# The graph shows how default payment varies by marital status:
# Both married and single clients make up the majority of the dataset.
# The default rate is slightly higher among single clients than married ones.
# Clients in the “others” category (including previously uncategorized values) are very few but show a noticeable default proportion, indicating potential risk.


# Credit Limit Distribution by Default Status:
# The boxplot compares credit limits between clients who defaulted and those who didn’t:
# Clients who did not default generally have higher median credit limits.
# Clients who defaulted tend to have lower credit limits, though there are a few high-limit clients who also defaulted (shown as outliers).
# This indicates that credit limit may be a significant factor in predicting default risk — those with lower limits are more likely to default.


#%%
# Age vs. default behavior:
df['Age_Group'] = pd.cut(df['Age'], bins=[20, 30, 40, 50, 60, 100], 
                         labels=['21–30', '31–40', '41–50', '51–60', '60+'])
age_default_rate = df.groupby('Age_Group')['Default_Payment'].mean()
print(age_default_rate)

import matplotlib.pyplot as plt

age_default_rate.plot(kind='bar', color='skyblue')
plt.title('Default Rate by Age Group')
plt.ylabel('Default Rate')
plt.xlabel('Age Group')
plt.ylim(0, 0.3)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
#%%


#%%
# Interpretation:
# older clients (60+) actually have the highest default rate in this dataset at 26.8%.
# The lowest default rate is among clients aged 31–40.
# While the youngest group (21–30) shows a moderately high rate (22.4%), the risk increases gradually with age.
# Younger clients are not the most likely to default. 
# In this dataset, default risk increases with age, particularly among clients over 50.
#%%
# 


#%% 
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set(style="whitegrid")

# Boxplots: Bill Amounts by Default Status
bill_cols = ['Bill_Amount_Sep', 'Bill_Amount_Aug', 'Bill_Amount_Jul',
             'Bill_Amount_Jun', 'Bill_Amount_May', 'Bill_Amount_Apr']

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
for i, col in enumerate(bill_cols):
    sns.boxplot(x='Default_Payment', y=col, data=df, ax=axes[i // 2, i % 2])
    axes[i // 2, i % 2].set_title(f'{col} by Default Status')
    axes[i // 2, i % 2].set_xticklabels(['No Default', 'Default'])

plt.tight_layout()
plt.show()
#%%


#%%
# 2. Boxplots: Payment Amounts by Default Status
payment_cols = ['Payment_Amount_Sep', 'Payment_Amount_Aug', 'Payment_Amount_Jul',
                'Payment_Amount_Jun', 'Payment_Amount_May', 'Payment_Amount_Apr']

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
for i, col in enumerate(payment_cols):
    sns.boxplot(x='Default_Payment', y=col, data=df, ax=axes[i // 2, i % 2])
    axes[i // 2, i % 2].set_title(f'{col} by Default Status')
    axes[i // 2, i % 2].set_xticklabels(['No Default', 'Default'])

plt.tight_layout()
plt.show()
#%%


#%%
# 3. Momentum Calculation
def calculate_weighted_momentum(row):
    pay_status_cols = ['Repayment_Status_Sep', 'Repayment_Status_Aug', 'Repayment_Status_Jul',
                       'Repayment_Status_Jun', 'Repayment_Status_May', 'Repayment_Status_Apr']
    weights = [6, 5, 4, 3, 2, 1]
    weighted_sum = sum(weights[i] * row[pay_status_cols[i]] for i in range(len(pay_status_cols)))
    return "Bad Momentum" if weighted_sum > 20 else "Stable/Improving"

# Apply momentum if not already calculated
if 'Momentum' not in df.columns:
    df['Momentum'] = df.apply(calculate_weighted_momentum, axis=1)

print(df['Momentum'].value_counts())
#%%


#%%
# 4. Momentum vs Default Payment
plt.figure(figsize=(8, 6))
sns.countplot(x='Momentum', hue='Default_Payment', data=df, palette='Set2')
plt.title('Default Payment Distribution by Momentum')
plt.xlabel('Momentum Status')
plt.ylabel('Client Count')
plt.legend(title='Default Payment', labels=['No Default', 'Default'])
plt.show()
#%%


#%%
# 5. Momentum and Default Rate
momentum_default_rate = df.groupby('Momentum')['Default_Payment'].mean()
print("Default Rates by Momentum Category:\n", momentum_default_rate)
#%%


# %%
# Interpretation 
# Bill Amounts by Default Status (boxplot):
# These boxplots compare the monthly bill amounts for clients who defaulted versus those who did not across six months (April–September).
# Non-defaulters generally show higher variability in bill amounts, possibly reflecting more active or diverse credit usage.
# Median bill amounts are similar for both groups, meaning the total billed isn’t necessarily a strong standalone indicator of default.
# The presence of outliers among both groups suggests a wide range of credit behaviors, but bill amount alone is not strongly predictive of default status.


# Payment Amounts by Default Status:
# These boxplots show the distribution of payment amounts for defaulters and non-defaulters over the same six-month period.
# Defaulters tend to make significantly lower payments, with many making little to no payments each month.
# Non-defaulters consistently pay more, indicating stronger financial responsibility and a lower likelihood of default.
# These plots confirm that low monthly payments are strongly associated with future default, making payment behavior a critical risk factor.


# Momentum Calculation:
# Momentum was calculated using a weighted sum of six months of repayment status, assigning greater importance to recent months. 
# Clients whose weighted sum exceeded 20 were labeled as having “Bad Momentum”, reflecting deteriorating financial behavior. 
# Those with lower sums were labeled “Stable/Improving”, indicating good or improving habits.
# This approach helped uncover hidden trends in behavior, which turned out to be one of the strongest predictors of default in the dataset.


# Momentum vs. Default Payment:
# This bar plot visualizes the relationship between momentum (based on payment trends) and default rates.
# Clients with “Bad Momentum” (worsening or consistently poor payment status) show a much higher rate of default.
# Clients with “Stable/Improving Momentum” are far less likely to default.
# This confirms that momentum is a powerful behavioral signal, tracking trends over time is more insightful than one-time values.


# Default Rates by Momentum Category:
# This calculation shows the average default rate for each momentum group:
# Momentum Status	    Default Rate:
# Bad Momentum	     -     64.1%
# Stable/Improving	 -    17.0%

# The default rate is nearly 4 times higher among clients with Bad Momentum.
# This reinforces the value of momentum as a highly predictive feature in credit risk models.

#%% 
# Calculate average bill and payment amounts over 6 months
bill_avg = df[['Bill_Amount_Sep', 'Bill_Amount_Aug', 'Bill_Amount_Jul',
               'Bill_Amount_Jun', 'Bill_Amount_May', 'Bill_Amount_Apr']].mean(axis=1)

payment_avg = df[['Payment_Amount_Sep', 'Payment_Amount_Aug', 'Payment_Amount_Jul',
                  'Payment_Amount_Jun', 'Payment_Amount_May', 'Payment_Amount_Apr']].mean(axis=1)

# Compute the payment-to-bill ratio
df['Payment_Bill_Ratio'] = payment_avg / bill_avg

# Properly handle infinities and NaNs without chained assignment warnings
df['Payment_Bill_Ratio'] = df['Payment_Bill_Ratio'].replace([float('inf'), -float('inf')], 0)
df['Payment_Bill_Ratio'] = df['Payment_Bill_Ratio'].fillna(0)
print(df[['Client_ID', 'Payment_Bill_Ratio']].head())
#%%

#%%
# Interpretation:
# Payment-to-Bill Ratio, which captures the repayment behavior of clients over six months.
# It measures how much, on average, a client pays compared to what they owe.
# A low ratio (e.g., below 0.2) means a client is paying back less than 20% of their billed amount, which can be a sign of financial stress or risk of default.
# A high ratio (close to or above 1.0) indicates strong repayment discipline.



#%%
# Correlation matrix
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = df[numerical_cols].corr()

# Plot heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap Including Payment-to-Bill Ratio')
plt.tight_layout()
plt.show()
#%%

#%%
# Interpretation:
# The heatmap provides a visual overview of how numerical features relate to one another.
# It shows clusters of highly correlated features, such as:
# Monthly bill amounts are highly correlated with each other.
# Monthly payment amounts show similar patterns.
# It also highlights features that are moderately or weakly related to Default_Payment.


#%%
# Extract top correlated features
corr_with_default = corr_matrix['Default_Payment'].drop('Default_Payment')
sorted_corr = corr_with_default.abs().sort_values(ascending=False)
print("Top correlations with Default_Payment:\n", sorted_corr.head(10))
# %%

#%%
# Interpretation Summary:
# Repayment status across all six months is by far the most informative category of features.
# The more recent the repayment delay, the stronger the correlation with default.
# Credit limit also correlates: clients with higher limits are generally more creditworthy.
# Recent payment amounts show weaker correlations, but still highlight behavioral patterns, especially when combined with momentum or ratio metrics.

# END OF EDA