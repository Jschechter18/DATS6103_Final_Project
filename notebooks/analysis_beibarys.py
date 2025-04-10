#%%
# import dataset 
import pandas as pd 
df = pd.read_csv("hf://datasets/scikit-learn/credit-card-clients/UCI_Credit_Card.csv")


#### Cleaning #####
# %%
# few rows of the data set, head and tail 
print(f"Top 5 rows of the dataset:\n{df.head(5)}\n")
print(f"Least 5 rows of the dataset:\n{df.tail(5)}\n")
# %%
# information about the dataset 
print(f"Information about the dataset:{df.info()}\n")

# %% 
# change the columns names for more comfortable usage:
df.columns = ["id", 
              "credit_limit", 
              "sex", 
              "education", 
              "marriage_status", 
              "age", 
              "repayment_september", 
              "repayment_august", 
              "repayment_july", 
              "repayment_june", 
              "repayment_may", 
              "repayment_april", 
              "bill_september", 
              "bill_august", 
              "bill_july", 
              "bill_june", 
              "bill_may", 
              "bill_april",
              "payment_am_september",
              "payment_am_august",
              "payment_am_july",
              "payment_am_june",
              "payment_am_may",
              "payment_am_april",
              "default_status"]

# terminate id as we don't need it 
df.drop("id", inplace = True, axis = 1)

# columns after the change: 
print(f"Columns of the dataset after the change:\n{df.info()}\n")

# %% 
# unique values of sex 
df["sex"].unique()

# 1 - female 
# 2 - male 
# Change accordingly 
df["sex"] = df["sex"].map({1: "Male", 2: "Female"})

# after 
df["sex"].unique()

# %% 
# unique values of marriage_status 
df["marriage_status"].unique()

# 1 - married 
# 2 - single 
# 3 - others 
df["marriage_status"] = df["marriage_status"].map({1 : "Married", 
                                                  2 : "Single", 
                                                  3 : "Other", 
                                                  0 : "Other"})

# after 
df["marriage_status"].unique()


# %% 
# Describe 
df.describe()

# %% 
# repayment columns 
repayment_cols = [col for col in df.columns if "repay" in col]

for col in repayment_cols:
    print(f"{col}: {sorted(df[col].unique())}")

# %% 
# Credit Limit 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Credit Limit Distribution
plt.figure(figsize= (12, 10))

sns.histplot(data = df, 
             x = "credit_limit")

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
sns.boxplot(data = df, 
            y = "default_status",
             x = "credit_limit", 
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

sns.boxplot(data = df, 
            y = "default_status",
             x = "credit_limit", 
             orient= "h", 
             hue = "sex")

# design 
plt.xlabel("Credit Limit", 
           fontdict= {"fontsize": 20})

plt.ylabel("Default Status", 
           fontdict= {"fontsize": 20})

plt.tick_params(axis='both', labelsize=15)

plt.subplot(1, 2, 2)
sns.boxplot(data = df, 
            y = "default_status",
             x = "credit_limit", 
             orient= "h", 
             hue = "marriage_status")

# design 
plt.xlabel("Credit Limit", 
           fontdict= {"fontsize": 20})

plt.ylabel("Default Status", 
           fontdict= {"fontsize": 20})

plt.tick_params(axis='both', labelsize=15)

plt.show()

# %% 
# Stacked Histogram on sex and marriage - default status 
plt.figure(figsize= (12, 10))

plt.subplot(1, 2, 1)
ax = sns.histplot(data = df, 
             x = "sex", 
             hue = "default_status", 
             multiple= "fill")

# design 
plt.xlabel("Sex",  
           fontdict= {"fontsize": 20})

plt.ylabel("Frequency", 
           fontdict= {"fontsize": 20})

plt.tick_params(axis='both', labelsize=15)

for container in ax.containers:
    labels = [f'{h.get_height()*100:.1f}%' if h.get_height() > 0 else '' for h in container]
    ax.bar_label(container, labels=labels, label_type='center', fontsize=12)

plt.subplot(1, 2, 2)
ax = sns.histplot(data = df, 
             x = "marriage_status", 
             hue = "default_status", 
             multiple= "fill")

for container in ax.containers:
    labels = [f'{h.get_height()*100:.1f}%' if h.get_height() > 0 else '' for h in container]
    ax.bar_label(container, labels=labels, label_type='center', fontsize=12)

# design 
plt.xlabel("Marriage Status",  
           fontdict= {"fontsize": 20})

plt.ylabel(None)

plt.tick_params(axis='both', labelsize=15)

plt.show()




# %%
# distribution of Age 
plt.figure(figsize= (12, 10))

sns.histplot(data = df, 
             x = "age")

# design 
plt.title("Overall distribution of Age", 
          fontdict= {"fontsize":30, 
                     "fontweight":"bold"})
plt.xlabel("Age", 
           fontdict= {"fontsize": 20})

plt.ylabel("Frequency", 
           fontdict= {"fontsize": 20})

plt.tick_params(axis='both', labelsize=15)

plt.show()

# age vs default status 
plt.figure(figsize= (12, 10))
sns.boxplot(data = df, 
            y = "default_status",
             x = "age", 
             orient= "h")

# design 
plt.title("Overall distribution of Age by Default Status", 
          fontdict= {"fontsize":30, 
                     "fontweight":"bold"})
plt.xlabel("Age", 
           fontdict= {"fontsize": 20})

plt.ylabel("Default Status", 
           fontdict= {"fontsize": 20})

plt.tick_params(axis='both', labelsize=15)

plt.show()

# Age and Marriage Status vs Default Status 
plt.figure(figsize= (12, 10))

plt.subplot(1, 2, 1)

sns.boxplot(data = df, 
            y = "default_status",
             x = "age", 
             orient= "h", 
             hue = "sex")

# design 
plt.xlabel("Age", 
           fontdict= {"fontsize": 20})

plt.ylabel("Default Status", 
           fontdict= {"fontsize": 20})

plt.tick_params(axis='both', labelsize=15)

plt.subplot(1, 2, 2)
sns.boxplot(data = df, 
            y = "default_status",
             x = "age", 
             orient= "h", 
             hue = "marriage_status")

# design 
plt.xlabel("Age", 
           fontdict= {"fontsize": 20})

plt.ylabel("Default Status", 
           fontdict= {"fontsize": 20})

plt.tick_params(axis='both', labelsize=15)

plt.show()

# %% 
# Education and Marriage Statuses 
# Change integers to Actual Levels 
# 1 - graduate
# 2 - undergraduate
# 3 - high school
# anything else - other 
print(f"Unique values of Education Column:\n{df['education'].unique()}\n")

# change 
df["education"] = df["education"].map({1 : "Graduate", 
                                                  2 : "Undergraduate", 
                                                  3 : "High School"}).fillna("Other")

# check the change:
print(f"Unique values of Education Column aft the change:\n{df['education'].unique()}\n")

# %% 
# Education and Mariage Statuses Visualizations 
plt.figure(figsize= (15, 12))

plt.subplot(1, 2, 1)
ax = sns.histplot(data = df, 
             y = "education", 
             hue = "default_status", 
             multiple= "fill")

# design 
plt.xlabel("Education",  
           fontdict= {"fontsize": 20})

plt.ylabel("Frequency", 
           fontdict= {"fontsize": 20})

plt.tick_params(axis='both', labelsize=15)

for container in ax.containers:
    # Because in "fill" mode, heights are already fractions (like 0.4, 0.6)
    labels = [f'{h.get_width()*100:.1f}%' if h.get_width() > 0 else '' for h in container]
    ax.bar_label(container, labels=labels, label_type='center', fontsize=12)

plt.show()

# %%
# Repayment Status vs. Default Visualizations 
# columns
columns = ["repayment_september", 
              "repayment_august", 
              "repayment_july", 
              "repayment_june", 
              "repayment_may", 
              "repayment_april"]

# count plots for each column
# figure size
# (width, height)
plt.figure(figsize=(30, 12))

# columns
for i, col in enumerate(columns, 1):
    plt.subplot(2, 4, i)
    ax = sns.countplot(x=col, hue='default_status', data=df, palette='Set2')
    # title
    plt.title(f'{col} Distribution by Default Status', fontweight = "bold", fontsize = 18)
    # xlabel
    plt.xlabel(f'{col} status', fontsize = 14)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)

    # ylabel
    plt.ylabel('Count', fontsize = 14)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize = 14)

    # legend
    plt.legend(title = "Default Status",
               title_fontsize = 18,
               fontsize = 14)


plt.tight_layout()
plt.show()
# %%
# proportions
# figure size
plt.figure(figsize=(30, 20))

# columns
for i, col in enumerate(columns, 1):
    plt.subplot(2, 4, i)
    ax = sns.histplot(x=col, hue='default_status', data=df, multiple='fill', palette='Set2')
  
    # xlabel
    plt.xlabel(f'{col} Status', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
    
    # ylabel
    plt.ylabel('Proportion', fontsize=14)  
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)

    # legend
    plt.legend(title="Default Status", title_fontsize=18, fontsize=14)

plt.show()

# %%
# bill amounts and payment amounts 

# bill columns 
bill_columns = ["bill_september", "bill_august", "bill_july", "bill_june", "bill_may", "bill_april"]

# payment columns
payment_columns = ["payment_am_september", "payment_am_august", "payment_am_july", "payment_am_june", "payment_am_may", "payment_am_april"]

# Calculate the average of bills and payments
df['average_bill'] = df[bill_columns].mean(axis=1)
df['average_payment'] = df[payment_columns].mean(axis=1)

# The ratio of average payment to average bill
df['payment_to_bill_ratio'] = df['average_payment'] / df['average_bill']

# %% 
# bill visualizations by marriage and education
plt.figure(figsize= (12, 10))

plt.subplot(1, 2, 1)

sns.boxplot(data = df, 
            y = "default_status",
             x = "average_bill", 
             orient= "h", 
             hue = "marriage_status")

# design 
plt.xlabel("Average Bill Amount", 
           fontdict= {"fontsize": 20})

plt.ylabel("Default Status", 
           fontdict= {"fontsize": 20})

plt.tick_params(axis='both', labelsize=15)

plt.subplot(1, 2, 2)
sns.boxplot(data = df, 
            y = "default_status",
             x = "average_bill", 
             orient= "h", 
             hue = "education")

# design 
plt.xlabel("Average Bill Amount", 
           fontdict= {"fontsize": 20})

plt.ylabel('', 
           fontdict= {"fontsize": 20})

plt.tick_params(axis='both', labelsize=15)

plt.show()
# %%
# payment visualizations by marriage 
plt.figure(figsize= (12, 10))

plt.subplot(1, 2, 1)

sns.boxplot(data = df, 
            y = "default_status",
             x = "average_payment", 
             orient= "h", 
             hue = "marriage_status")

# design 
plt.xlabel("Average Payment Amount", 
           fontdict= {"fontsize": 20})

plt.ylabel("Default Status", 
           fontdict= {"fontsize": 20})

plt.tick_params(axis='both', labelsize=15)

plt.subplot(1, 2, 2)
sns.boxplot(data = df, 
            y = "default_status",
             x = "average_payment", 
             orient= "h", 
             hue = "education")

# design 
plt.xlabel("Average Payment Amount", 
           fontdict= {"fontsize": 20})

plt.ylabel('', 
           fontdict= {"fontsize": 20})

plt.tick_params(axis='both', labelsize=15)

plt.show()
# %%
# Select only numeric columns
numeric_cols = df.select_dtypes(include='number')

# Pearson correlation
pearson_corr = numeric_cols.corr(method='pearson')['default_status']

# Sort and plot
correlations_with_target = pearson_corr.reindex(pearson_corr.abs().sort_values(ascending=False).index)

# Plot
plt.figure(figsize=(12, 10))
correlations_with_target.plot(kind='bar')
plt.title('Pearson Correlation of Features with Default Status')
plt.ylabel('Correlation Coefficient')
plt.xlabel('Features')
plt.show()




