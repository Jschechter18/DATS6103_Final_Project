
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# -----------------------------
# Data Preparation for statistical Testing
# -----------------------------

# Load your cleaned CSV
df = pd.read_csv("credit_card_clients_full.csv")

# Rename columns
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

# Clean category values
df['Education_Level'] = df['Education_Level'].replace({0: 4, 5: 4, 6: 4})
df['Marital_Status'] = df['Marital_Status'].replace({0: 3})


#%%
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


#%%
# Create Momentum column
def calculate_weighted_momentum(row):
    pay_cols = ['Repayment_Status_Sep', 'Repayment_Status_Aug', 'Repayment_Status_Jul',
                'Repayment_Status_Jun', 'Repayment_Status_May', 'Repayment_Status_Apr']
    weights = [6, 5, 4, 3, 2, 1]
    return "Bad Momentum" if sum(weights[i] * row[pay_cols[i]] for i in range(6)) > 20 else "Stable/Improving"

df['Momentum'] = df.apply(calculate_weighted_momentum, axis=1)
#%%


#%%
# Momentum Calculation:
# Momentum was calculated using a weighted sum of six months of repayment status, assigning greater importance to recent months. 
# Clients whose weighted sum exceeded 20 were labeled as having “Bad Momentum”, reflecting deteriorating financial behavior. 
# Those with lower sums were labeled “Stable/Improving”, indicating good or improving habits.
# This approach helped uncover hidden trends in behavior, which turned out to be one of the strongest predictors of default in the dataset.
#%%


#%%
# Create Payment-to-Bill Ratio
bill_cols = [f'Bill_Amount_{m}' for m in ['Sep', 'Aug', 'Jul', 'Jun', 'May', 'Apr']]
pay_cols = [f'Payment_Amount_{m}' for m in ['Sep', 'Aug', 'Jul', 'Jun', 'May', 'Apr']]

bill_avg = df[bill_cols].mean(axis=1)
pay_avg = df[pay_cols].mean(axis=1)
df['Payment_Bill_Ratio'] = pay_avg / bill_avg
df['Payment_Bill_Ratio'] = df['Payment_Bill_Ratio'].replace([np.inf, -np.inf], 0)
df['Payment_Bill_Ratio'] = df['Payment_Bill_Ratio'].fillna(0)


#%%
# Interpretation:
# Payment-to-Bill Ratio, which captures the repayment behavior of clients over six months.
# It measures how much, on average, a client pays compared to what they owe.
# A low ratio (e.g., below 0.2) means a client is paying back less than 20% of their billed amount, which can be a sign of financial stress or risk of default.
# A high ratio (close to or above 1.0) indicates strong repayment discipline.
#%% 


#%%
# Drop rows with missing values in required columns
required_cols = [
    'Momentum', 'Default_Payment', 'Education_Level', 'Marital_Status',
    'Credit_Limit', 'Payment_Bill_Ratio', 'Age'
]
df.dropna(subset=required_cols, inplace=True)

# Convert appropriate columns to categorical
cat_columns = ['Default_Payment', 'Momentum', 'Education_Level', 'Marital_Status']
for col in cat_columns:
    df[col] = df[col].astype('category')

# Create Age Group column for ANOVA
df['Age_Group'] = pd.cut(df['Age'], bins=[20, 30, 40, 50, 60, 100],
                         labels=['21–30', '31–40', '41–50', '51–60', '60+'])

# Display group sizes for chi-square test sanity check
print("\nGroup sizes:")
print("Momentum:\n", df['Momentum'].value_counts())
print("Education:\n", df['Education_Level'].value_counts())
print("Marital Status:\n", df['Marital_Status'].value_counts())
print("Age Groups:\n", df['Age_Group'].value_counts())


#%%
# Check distribution of a key numeric variable (e.g. credit limit)
sns.histplot(df['Credit_Limit'], kde=True)
plt.title('Distribution of Credit Limit')
plt.xlabel('Credit Limit')
plt.show()
# %%

#%%
# Interpretation:
# The histogram reveals that the distribution of credit limits is right-skewed, 
# meaning most clients have lower credit limits, while a few have very high limits (outliers). 
# The majority of credit limits fall between 50,000 and 200,000.
#%%



#%%
from scipy import stats

# Chi-Square test: Momentum vs Default
momentum_table = pd.crosstab(df['Momentum'], df['Default_Payment'])
chi2, p1, _, _ = stats.chi2_contingency(momentum_table)

# Print the result
print("\nMomentum vs Default")
print("Test Type: Chi-Square")
print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"p-value: {p1:.4f}")
#%%

#%%
# Chi-Square Test – Momentum vs Defaulting
# Hypothesis:
# H₀ : There is no relationship between momentum group and defaulting.
# Ha: There is a significant relationship between momentum group and defaulting.

# Chi-Square Statistic: χ² = 3730.0425
# p-value: p = 0.0000

# Interpretation:
# Since p < 0.05, we reject the null hypothesis.
# There is a statistically significant relationship between momentum status and defaulting.
# Clients with bad momentum (worsening payment history) are much more likely to default, supporting its use as a behavioral risk indicator.



#%%
# T-Test: Credit Limit vs Default
group1 = df[df['Default_Payment'] == 1]['Credit_Limit']
group2 = df[df['Default_Payment'] == 0]['Credit_Limit']

t_stat1, p2 = stats.ttest_ind(group1, group2, equal_var=False)

# Print the result
print("\nCredit Limit vs Default")
print(f"T-Statistic: {t_stat1:.4f}")
print(f"p-value: {p2:.4f}")


#%%

#%%
# T-Test – Credit Limit vs Defaulting
# Hypotheses:
# H₀: The average credit limit is the same for defaulters and non-defaulters.
# Ha: There is a significant difference in credit limit between the two groups

# T-Statistic: t = -28.9516
# p-value: p = 0.0000

# Interpretation:
# Since p < 0.05, we reject the null hypothesis.
# There is a significant difference in credit limits between defaulters and non-defaulters.
# Defaulters tend to have lower credit limits, indicating that credit limit can help discriminate risk levels.


#%%
# T-Test: Payment-to-Bill Ratio vs Default
group3 = df[df['Default_Payment'] == 1]['Payment_Bill_Ratio']
group4 = df[df['Default_Payment'] == 0]['Payment_Bill_Ratio']

t_stat2, p3 = stats.ttest_ind(group3, group4, equal_var=False)
# Print the result
print("\nPayment-to-Bill Ratio vs Default")
print(f"T-Statistic: {t_stat2:.4f}")
print(f"p-value: {p3:.4f}")
#%%

#%%
# T-Test: Payment-to-Bill Ratio vs Defaulting
# Hypotheses:
# H₀: There is no difference in average payment-to-bill ratio between defaulters and non-defaulters.
# Ha: There is a significant difference in payment-to-bill ratio between the two groups.

# T-Statistic: t = -2.1546
# p-value: p = 0.0000

# Interpretation:
# Since p < 0.05, we reject the null hypothesis.
# The average Payment-to-Bill Ratio differs significantly between defaulters and non-defaulters.
# Defaulters pay back much less of their bills on average, confirming this is a valuable behavioral metric for predicting default.




#%%
# Chi-Square: Education vs Default
edu_ct = pd.crosstab(df['Education_Level'], df['Default_Payment'])

chi2_edu, p4, _, _ = stats.chi2_contingency(edu_ct)
# Print the result
print("\n Education Level vs Default")
print(f"Chi-Square Statistic: {chi2_edu:.4f}")
print(f"p-value: {p4:.4f}")
#%%

#%%
# Chi-Square Test: Education Level vs Defaulting
# Hypotheses:
# H₀: Defaulting is independent of education level.
# Ha: Education level and defaulting are associated.

# Chi-Square Statistic: χ² = 160.4100
# p-value: p = 0.0000

# Interpretation:
# Since p < 0.05, we reject the null hypothesis.
# Education level is statistically associated with defaulting.
# Lower education levels (e.g., high school or others) are generally linked with higher risk of default.




#%%
# Chi-Square: Marital Status vs Default
marital_ct = pd.crosstab(df['Marital_Status'], df['Default_Payment'])

chi2_mar, p5, _, _ = stats.chi2_contingency(marital_ct)

# Print the result
print("\n Marital Status vs Default")
print(f"Chi-Square Statistic: {chi2_mar:.4f}")
print(f"p-value: {p5:.4f}")
#%%

#%%
# Chi-Square Test: Marital Status vs Defaulting
# Hypotheses:
# H₀: Marital status is independent of defaulting.
# Ha: Marital status is associated with defaulting.

# Chi-Square Statistic: χ² = 28.1303
# p-value: p = 0.0016

# Interpretation:
# Since p < 0.05, we reject the null hypothesis.
# Marital status has a significant relationship with defaulting.
# In this dataset, single clients tend to default more often than married ones.




#%%
# ANOVA: Age Group vs Default
anova_groups = [group['Default_Payment'].values for _, group in df.groupby('Age_Group', observed=False)]
f_stat, p6 = stats.f_oneway(*anova_groups)

# Print the result
print("\n Age Group vs Default")
print(f"F-Statistic: {f_stat:.4f}")
print(f"p-value: {p6:.4f}")
# %%


#%%
# ANOVA: Age Group vs Defaulting
# Hypotheses:
# H₀: All age groups have the same mean default rate.
# Ha: At least one age group has a significantly different default rate.

# F-Statistic: F = 9.4988
# p-value: p = 0.00

# Interpretation:
# Since p < 0.05, we reject the null hypothesis.
# There is a significant difference in default rates among age groups.
# Contrary to expectations, older clients (e.g., 60+) had the highest default rates, highlighting the need to reconsider age-based assumptions in credit risk.


# Conclusion:
# All statistical tests revealed significant relationships between defaulting and various demographic and behavioral feature
# Momentum, Credit Limit, and Payment-to-Bill Ratio are strong predictors of default.
# Education, Marital Status, and Age Group also show meaningful differences in default risk.



