#%%
from datasets import load_dataset
import pandas as pd

# Load the dataset
dataset = load_dataset("scikit-learn/credit-card-clients", split='train')

# Convert to a pandas DataFrame
df = dataset.to_pandas()

# Save the dataset to CSV
df.to_csv("credit_card_clients_full.csv", index=False)

# Display the number of rows and the first few entries
print(f"Total rows: {len(df)}")
print(df.head())

# Check for missing values
print("\n Missing values per column:")
print(df.isnull().sum())
# %%



#%%
import matplotlib.pyplot as plt
import seaborn as sns

# Clean column names
df.columns = [col.strip().lower().replace('.', '_').replace(' ', '_') for col in df.columns]
df.rename(columns={'default_payment_next_month': 'default'}, inplace=True)

# Preprocess 'education' and 'marriage'
df['education'] = df['education'].replace({0: 4, 5: 4, 6: 4})  # Group unknown education into '4'
df['marriage'] = df['marriage'].replace({0: 3})  # Group unknown marriage into '3'

# Create average payment amount feature
df['avg_pay_amt'] = df[[f'pay_amt{i}' for i in range(1, 7)]].mean(axis=1)

print(df.columns.tolist())
df.head()

#%%
#Research Question 1: What features best predict payment defaults?
# We'll start by plotting a correlation heatmap of numeric feature
plt.figure(figsize=(16, 12))
sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap: Numeric Features vs Default", fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# EDA 1: Target class distribution
sns.countplot(x='default', data=df)
plt.title("Default Class Distribution")
plt.xlabel("Default (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()


#%%
# Research Question 2: How do education and marriage status relate to default?
# Education vs Default
sns.countplot(x='education', hue='default', data=df)
plt.title("Education Level vs Default")
plt.xlabel("Education (1=Grad, 2=University, 3=High School, 4=Others)")
plt.ylabel("Count")
plt.legend(title="Default")
plt.show()

# Marriage vs Default
sns.countplot(x='marriage', hue='default', data=df)
plt.title("Marriage Status vs Default")
plt.xlabel("Marriage (1=Married, 2=Single, 3=Others)")
plt.ylabel("Count")
plt.legend(title="Default")
plt.show()


# Research Question 3: Are younger clients more likely to default?
# Age distribution by Default
sns.histplot(data=df, x='age', hue='default', multiple='stack', bins=30, kde=True)
plt.title("Age Distribution by Default")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()


# Research Question 4: Does higher credit limit reduce default likelihood?
# Credit Limit vs Default
sns.boxplot(x='default', y='limit_bal', data=df)
plt.title("Credit Limit by Default")
plt.xlabel("Default")
plt.ylabel("Credit Limit")
plt.show()


# Research Question 5: Do minimum-payment behaviors increase default probability?
# Avg Payment Amount vs Default
sns.boxplot(x='default', y='avg_pay_amt', data=df)
plt.title("Average Monthly Payment vs Default")
plt.xlabel("Default")
plt.ylabel("Average Payment Amount")
plt.show()
# %%
