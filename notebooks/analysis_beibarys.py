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
# Convert Taiwanese currency to USD dollars
# Conversion rate 
conversion_rate = 1 / 31.8

# Columns that need to be converted 
nt_dollar_columns = [
    "credit_limit", 
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
              "payment_am_april"]

# Convert 
for col in nt_dollar_columns:
    df[col] = df[col] * conversion_rate

# Verify the conversion
df[nt_dollar_columns].head()


# View the resulting dataframe
df.head()

# %% 

# %% 
### VISUALIZATIONS ###
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
# Calculate Momentum Feature 
# Select Needed variables 
pay_status_cols = [
     "repayment_april",
     "repayment_may", 
     "repayment_june", 
     "repayment_july",
    "repayment_august", 
    "repayment_september"]

# Formula for momentum 
def calculate_weighted_momentum(row):
    weights = [1, 2, 3, 4, 5, 6]  # Weights for April to September, increasing the closer to October
    weighted_sum = sum(weights[i] * row[pay_status_cols[i]] for i in range(len(pay_status_cols)))
    # return "Bad Momentum" if weighted_sum > 20 else "Stable/Improving"  # Adjust threshold as needed
    # 0 is good, 1 is bad
    return 1 if weighted_sum > 20 else 0  # Adjust threshold as needed

# Create momentum feature 
df["momentum"] = df.apply(calculate_weighted_momentum, axis=1)
df["momentum_label"] = df["momentum"].map({0: "Stable/Improving", 1: "Bad Momentum"})

# Check 
df["momentum_label"]

# %% 
# Momentum Visualization 
plt.figure(figsize=(12, 10))

# Create a count plot
ax = sns.histplot(
    data=df,
    x='momentum_label',
    hue='default_status', 
    multiple = "fill", 
    stat = "probability", 
    discrete= True  
)

# Title and labels
plt.title('Proportions of Momentum Classes by Default Status', fontsize=22, fontweight='bold')

# x axis 
plt.xlabel("Momentum Class", fontsize = 20)
plt.xticks(fontsize = 15)\


# yaxis 
plt.ylabel("Proportion", fontsize = 20)
plt.yticks(fontsize = 15)

# labels 
for c in ax.containers:
    labels = [f'{v.get_height():.2f}' if v.get_height() > 0 else '' for v in c]
    ax.bar_label(
        c, 
        labels=labels, 
        label_type='center', 
        fontsize=20, 
        weight='bold'
    )

plt.show()

# %% 
# Second plot: Momentum vs Default Status, split by Education Status
plt.figure(figsize=(12, 10))

ax1 = sns.displot(
    data=df,
    x='momentum_label',
    hue='default_status',
    multiple="fill",
    stat="probability",
    discrete=True,
    col='education',  
)

# labels 
for row in ax1.axes.flat:   
    for container in row.containers:
        labels = [f'{bar.get_height():.2f}' if bar.get_height() > 0 else '' for bar in container]
        row.bar_label(
            container,
            labels=labels,
            label_type='center',
            fontsize=15,
            weight='bold'
        )

# design 

plt.show()

# %% 
# Third plot: Momentum vs Default Status, split by Marriage Status
plt.figure(figsize=(12, 10))

ax1 = sns.displot(
    data=df,
    x='momentum_label',
    hue='default_status',
    multiple="fill",
    stat="probability",
    discrete=True,
    col='marriage_status',  
)

for row in ax1.axes.flat:   
    for container in row.containers:
        labels = [f'{bar.get_height():.2f}' if bar.get_height() > 0 else '' for bar in container]
        row.bar_label(
            container,
            labels=labels,
            label_type='center',
            fontsize=15,
            weight='bold'
        )

plt.show()

# %%
# Select only numeric columns
numeric_cols = df.select_dtypes(include='number')

# Pearson correlation
pearson_corr = numeric_cols.corr()['default_status']

# Sort and plot
correlations_with_target = pearson_corr.reindex(pearson_corr.abs().sort_values(ascending=False).index)

# Plot
plt.figure(figsize=(12, 10))
correlations_with_target.plot(kind='bar')

# Design 
# title 
plt.title('Pearson Correlation of Features with Default Status')

# y-axis
plt.ylabel('Correlation Coefficient')

# x-axis
plt.xlabel('Features')


plt.show()

# print the summary 
print(f"Correlations with Default Status:\n{correlations_with_target}\n")

# %% 
### New Features ### 
# average_status 
# columns 
rp_columns = ["repayment_september", 
              "repayment_august", 
              "repayment_july", 
              "repayment_june", 
              "repayment_may", 
              "repayment_april"]

df["average_repayment_status"] = df[rp_columns].mean(axis = 1)

# Check 
plt.figure(figsize=(12, 10))

sns.boxplot(
    data=df,
    y='default_status', 
    x='average_repayment_status', 
    orient = "h"
)

# Deisign
# title
plt.title('Average Repayment Status by Default Status', fontsize=22, fontweight='bold')

# y label
plt.ylabel('Default Status', fontsize=18)
plt.yticks(fontsize=14)

# x label
plt.xlabel('Average Repayment Status', fontsize=18)
plt.xticks(fontsize=14)

plt.show()

# %% 
# Repayment Volatility 
df['repayment_volatility'] = df[rp_columns].std(axis = 1)

# plot again 
plt.figure(figsize=(12, 10))

sns.boxplot(
    data=df,
    y='default_status', 
    x='repayment_volatility', 
    orient = "h"
)

# Deisign
# title
plt.title('Repayment Volatility by Default Status', fontsize=22, fontweight='bold')

# y label
plt.ylabel('Default Status', fontsize=18)
plt.yticks(fontsize=14)

# x label
plt.xlabel('Repayment Volatility', fontsize=18)
plt.xticks(fontsize=14)

plt.show()

# %% 
# Repayment Recent Mean Feature 
recent_months = ["repayment_july", "repayment_august", "repayment_september"]
df["recent_repayment_mean"] = df[recent_months].mean(axis=1)

# plot again 
plt.figure(figsize=(12, 10))

sns.boxplot(
    data=df,
    y='default_status', 
    x='recent_repayment_mean', 
    orient = "h"
)

# Deisign
# title
plt.title('Repayment Status Mean in 3 months by Default Status', fontsize=22, fontweight='bold')

# y label
plt.ylabel('Default Status', fontsize=18)
plt.yticks(fontsize=14)

# x label
plt.xlabel('Repayment Status Mean', fontsize=18)
plt.xticks(fontsize=14)

plt.show()

# %% 
# Momentum and Volatility
df["momentum_volatility_interaction"] = df["momentum"] * df["repayment_volatility"]

# plot again 
plt.figure(figsize=(12, 10))

sns.boxplot(
    data=df,
    y='default_status', 
    x='momentum_volatility_interaction', 
    orient = "h"
)

# Deisign
# title
plt.title('Momentum and Volatility by Default Status', fontsize=22, fontweight='bold')

# y label
plt.ylabel('Default Status', fontsize=18)
plt.yticks(fontsize=14)

# x label
plt.xlabel('Momentum and Volatility', fontsize=18)
plt.xticks(fontsize=14)

plt.show()

# %% 
# Number of Low Repayment Status months 
df["low_repayment_months"] = (df[rp_columns] > 1).sum(axis=1)

# plot again 
plt.figure(figsize=(12, 10))

sns.boxplot(
    data=df,
    y='default_status', 
    x='low_repayment_months', 
    orient = "h"
)

# Deisign
# title
plt.title('Number of Low Repayment months by Default Status', fontsize=22, fontweight='bold')

# y label
plt.ylabel('Default Status', fontsize=18)
plt.yticks(fontsize=14)

# x label
plt.xlabel('Number of Low Repayment months', fontsize=18)
plt.xticks(fontsize=14)

plt.show()

# %% 
# Risk Index
df["risk_index_1"] = df["momentum_volatility_interaction"] + 0.5 * df["low_repayment_months"]

# plot again 
plt.figure(figsize=(12, 10))

sns.boxplot(
    data=df,
    y='default_status', 
    x='risk_index_1', 
    orient = "h"
)

# Deisign
# title
plt.title('Risk Index by Default Status', fontsize=22, fontweight='bold')

# y label
plt.ylabel('Default Status', fontsize=18)
plt.yticks(fontsize=14)

# x label
plt.xlabel('Risk Index', fontsize=18)
plt.xticks(fontsize=14)

plt.show()

# %% 
# Super Default Score 
df["super_default_score"] = (
    0.5 * df["low_repayment_months"] +     # Strongest, but not too dominant
    0.35 * df["momentum"] +                 # Close to 0.4, important
    0.15 * df["repayment_volatility"]       # Still useful, but smaller
)

# plot again 
plt.figure(figsize=(12, 10))

sns.boxplot(
    data=df,
    y='default_status', 
    x='super_default_score', 
    orient = "h"
)

# Deisign
# title
plt.title('Super Default Score by Default Status', fontsize=22, fontweight='bold')

# y label
plt.ylabel('Default Status', fontsize=18)
plt.yticks(fontsize=14)

# x label
plt.xlabel('Super Default Score', fontsize=18)
plt.xticks(fontsize=14)

plt.show()

# %% 
df["super_default_score_v2"] = (
    0.45 * df["low_repayment_months"] +             # Strongest
    0.25 * df["momentum"] +                          # Second strongest
    0.15 * df["recent_repayment_mean"] +             # Recency matters
    0.10 * df["momentum_volatility_interaction"] +   # Trend + volatility
    0.05 * df["repayment_volatility"]                # Weakest
)

# plot again 
plt.figure(figsize=(12, 10))

sns.boxplot(
    data=df,
    y='default_status', 
    x='super_default_score', 
    orient = "h"
)

# Deisign
# title
plt.title('Super Default Score by Default Status', fontsize=22, fontweight='bold')

# y label
plt.ylabel('Default Status', fontsize=18)
plt.yticks(fontsize=14)

# x label
plt.xlabel('Super Default Score', fontsize=18)
plt.xticks(fontsize=14)

plt.show()

# %% 
# Repayment Deterioration &  Acceleration
df["repayment_deterioration"] = df["repayment_september"] - df["repayment_june"]
df["repayment_acceleration"] = (
    (df["repayment_september"] - df["repayment_august"]) -
    (df["repayment_august"] - df["repayment_july"])
)

# Momentum Recent Mean Interaction 
df['momentum_recent_mean_interaction'] = df['momentum'] * df['recent_repayment_mean']

# Volatility and deterioration interaction
# df['volatility_deterioration_interaction'] = df['repayment_volatility'] * df['repayment_deterioration']

# Momentum Stability Flag
df['momentum_stability_flag'] = np.where(
    (df['momentum'] == 1) & (df['repayment_volatility'] < 1.0), 1, 0
)


# %% 
# Final Formula 
df["super_default_score_final"] = (
    0.4 * df["low_repayment_months"] +              # Core: missed payments
    0.2 * df["momentum"] +                           # Trend: good or bad momentum
    0.1 * df["repayment_volatility"] +               # Instability risk
    0.1 * df["recent_repayment_mean"] +              # Recent repayment behavior
    0.1 * df["momentum_volatility_interaction"] +    # Combined volatility + momentum
    0.05 * df["repayment_deterioration"] +           # How much repayment worsened
    0.05 * df["repayment_acceleration"]              # Whether worsening accelerated
)

# plot again 
plt.figure(figsize=(12, 10))

sns.boxplot(
    data=df,
    y='default_status', 
    x='super_default_score_final', 
    orient = "h"
)

# Deisign
# title
plt.title('Super Default Score by Default Status', fontsize=22, fontweight='bold')

# y label
plt.ylabel('Default Status', fontsize=18)
plt.yticks(fontsize=14)

# x label
plt.xlabel('Super Default Score', fontsize=18)
plt.xticks(fontsize=14)

plt.show()


# %% 
# Check correlations again 
numeric_cols = df.select_dtypes(include='number')

# Pearson correlation
pearson_corr = numeric_cols.corr()['default_status'].drop('default_status')
positive_corr = pearson_corr[pearson_corr > 0].sort_values(ascending=False)

# Print
print(f"Correlations with Default Status:\n{correlations_with_target}\n")

# Plot 
plt.figure(figsize=(12, 8))
positive_corr.plot(kind='barh')

# Invert y-axis
plt.gca().invert_yaxis()

# Design
# Titlee
plt.title('Positive Pearson Correlations with Default Status (Highest to Lowest)', fontsize=18, fontweight='bold')

# X-axis
plt.xlabel('Correlation Coefficient', fontsize=14)
plt.xticks(fontsize=12)

# Y-axis
plt.ylabel('Features', fontsize=14)
plt.yticks(fontsize=12)

plt.show()

# %%
# Correlation Matrix
full_corr_matrix = numeric_cols.corr()

# Get features with positive correlation to default_status
positive_corr_features = full_corr_matrix['default_status'][full_corr_matrix['default_status'] > 0].index.tolist()

# Subset the correlation matrix dynamically
positive_corr_matrix = full_corr_matrix.loc[positive_corr_features, positive_corr_features]

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    positive_corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)

# Design
# Title
plt.title('Correlation Matrix: Only Positively Correlated Features with Default Status', fontsize=18, fontweight='bold')

# X-axis
plt.xticks(rotation=45, ha='right')

# Y-axis
plt.yticks(rotation=0)

plt.show()

# %% 
# drop useless column 
df.drop("payment_to_bill_ratio", axis = 1, inplace = True)
