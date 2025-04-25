#%% [markdown]
## Imports and Data Information #### 
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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, precision_score, f1_score, recall_score, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, pearsonr, shapiro
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import time

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

pay_status_cols =  [
    "April_Pay_status", "May_Pay_status", "June_Pay_status",
    "July_Pay_status", "August_Pay_status", "Sept_Pay_status"
]

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

def run_logistic_regression(df, columns, target="default_payment_next_month", plot_name=None, max_iter=1000, test_size=0.2):
    X = df[columns]
    y = df[target]

    # model = LogisticRegression(max_iter=max_iter)
    model = LogisticRegression(max_iter=max_iter, class_weight="balanced")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    model.fit(X_train, y_train)
    
    # y_pred = model.predict(X_test)
    
    # NOTE: Trying out this for y_pred to see if we can get better results
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    # threshold = 0.4  # Adjust as needed
    best_threshold = 0.5
    best_f1 = 0
    for t in np.arange(0.1, 0.9, 0.01):
        y_temp_pred = (y_pred_proba >= t).astype(int)
        f1 = f1_score(y_test, y_temp_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    # y_pred = (y_pred_proba >= threshold).astype(int)
    # print(f"Best threshold: {best_threshold}")
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    mse = mean_squared_error(y_test, y_pred)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    if plot_name != None:
        plot_logistic_regression_results(y_test, y_pred, y_pred_proba, plot_name)
        # # Check if LIMIT_BAL is linear
        # check_logit_linearity(model, X_test, columns[-1])
    
    return accuracy, precision, conf_matrix, recall, f1

def run_random_forest_classifier(df, columns, target="default_payment_next_month", test_size=0.2):
    X = df[columns]
    y = df[target]

    model = RandomForestClassifier(class_weight="balanced", random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("F1:", f1_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    return classification_report(y_test, y_pred, output_dict=True)

def print_model_results(accuracy, precision, conf_matrix, recall, f1, model_number):
    print(f"F1 Score {model_number}: {f1}")
    print(f"Recall {model_number}: {recall}")
    print(f"Precision {model_number}: {precision}")
    print(f"Accuracy {model_number}: {accuracy}")
    print(f"Confusion Matrix {model_number}:\n{conf_matrix}")
    print()
    
def plot_logistic_regression_results(y_test, y_pred, y_pred_proba, feature_name, model_name="Logistic Regression"):
    # Plot Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
    plt.title(f"{feature_name} - Confusion Matrix")
    # plt.title(f"{model_name} - Confusion Matrix")
    plt.show()

    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{feature_name} - ROC Curve")
    # plt.title(f"{model_name} - ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(f"../figures/{feature_name}_{model_name}_roc_curve.png")
    plt.show()

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
#%%
# It doesn't have an ordinal relationship as it is a nominal categorical variable.
# Let's convert that to binary
# That's fine now!

# Marriage doesn't have an ordinal relationship as it is a pure nominal categorical variable.
# Let's convert that to binary
# 1 - Married
# 2 - Single

# Binary 
# 1 - Married
# 0 - Not Married 
clients_data["MARRIAGE"] = clients_data["MARRIAGE"].astype("category")
clients_data["MARRIAGE"] = clients_data["MARRIAGE"].map({1: 1, 2: 0})
clients_data["MARRIAGE"].value_counts(normalize= True)
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
# Default Status (Dependent Variable)
# Figure size 
plt.figure(figsize= (12, 10))

# Figure 
sns.countplot(data = clients_data, 
              x = "default_payment_next_month")

# Design 
plt.title("Frequency  of Defaulting Classes", 
          fontdict= {"fontweight":"bold", 
                     "fontsize":24})

# X-axis 
plt.xlabel("Default Status", 
           fontdict= {"fontsize":20})

# y-axis 
plt.ylabel("Frequency", 
           fontdict= {"fontsize":20})

plt.tick_params(axis='both', labelsize=18)

# show
plt.show()

#%% 
# Age Distribution and Default Status 
plt.figure(figsize= (12, 10))

sns.histplot(data = clients_data, 
             x = "AGE")

# design 
plt.title("Overall distribution of Age", 
          fontdict= {"fontsize":24, 
                     "fontweight":"bold"})
plt.xlabel("Age", 
           fontdict= {"fontsize": 20})

plt.ylabel("Frequency", 
           fontdict= {"fontsize": 20})

plt.tick_params(axis='both', labelsize=18)

plt.show()

# Age effect on Default Status 
plt.figure(figsize= (12, 10))
sns.boxplot(data = clients_data, 
            y = "default_payment_next_month",
             x = "AGE", 
             orient= "h")

# design 
plt.title("Overall distribution of Age by Default Status", 
          fontdict= {"fontsize":24, 
                     "fontweight":"bold"})
plt.xlabel("Age", 
           fontdict= {"fontsize": 20})

plt.ylabel("Default Status", 
           fontdict= {"fontsize": 20})

plt.tick_params(axis='both', labelsize=18)

plt.show()

#%% 
# Create a variable witth only defaulters 
defaulters = clients_data[clients_data['default_payment_next_month'] == 1]

# Count the number of defaulters for each category 
marriage_default = defaulters["MARRIAGE"].value_counts().sort_index()

# Plot 
plt.figure(figsize= (12, 10))

# Marriage and Default Status 
sns.barplot(x = marriage_default.index,
             y = marriage_default.values)

# show 
plt.show()

#%%
# Marriage and Default 
# create a cross-tabulation for the Marraiage and Default 
proportions = pd.crosstab(clients_data['MARRIAGE'], 
                          clients_data['default_payment_next_month'], 
                          normalize='index').reset_index()

# Melt the data 
proportions_melted = proportions.melt(id_vars='MARRIAGE', 
                                      value_vars=[0, 1], 
                                      var_name='Default', 
                                      value_name='Proportion')

# Plot as grouped bar chart
plt.figure(figsize=(12, 10))

# plot itself
ax = sns.barplot(data=proportions_melted, 
            x='MARRIAGE', 
            y='Proportion', 
            hue='Default')

# Design
# title 
plt.title("Proportion of Default by Marriage Status", 
          fontweight = "bold", 
          fontsize=24)

# x-axis
plt.xlabel("Marriage Status", fontsize=20)
plt.xticks(ticks=[0, 1], labels=["Married", "Single"])

# y-axis
plt.ylabel("Proportion", fontsize=20)

plt.tick_params(axis='both', labelsize=18)

# Legend
plt.legend(title="Default Status", 
           title_fontsize = 15,
           fontsize = 15)

# labels for difference
for p in ax.patches:
    height = p.get_height()
    if height > 0.005:  
        ax.annotate(f"{height:.2%}",
                    (p.get_x() + p.get_width() / 2, height),
                    ha='center', va='bottom',
                    fontsize=15)
    
plt.show()

#%%
# Education and Default 
# create a cross-tabulation for the Education and Default 
proportions_education = pd.crosstab(clients_data['EDUCATION'], 
                          clients_data['default_payment_next_month'], 
                          normalize='index').reset_index()

# Melt the data 
proportions_melted_education = proportions_education.melt(id_vars='EDUCATION', 
                                      value_vars=[0, 1], 
                                      var_name='Default', 
                                      value_name='Proportion')

# Plot as grouped bar chart
plt.figure(figsize=(12, 10))

# plot itself
ax = sns.barplot(data=proportions_melted_education, 
            x='EDUCATION', 
            y='Proportion', 
            hue='Default')

# Design
# title 
plt.title("Proportion of Default by Education Level", 
          fontweight = "bold", 
          fontsize=24)

# x-axis
plt.xlabel("Education Level", fontsize=20)
plt.xticks(ticks=[0, 1, 2], labels=["Graduate", "Undergraduate", "High School"])

# y-axis
plt.ylabel("Proportion", fontsize=20)

plt.tick_params(axis='both', labelsize=18)

# Legend
plt.legend(title="Default Status", 
           title_fontsize = 15,
           fontsize = 15)

# labels for difference
for p in ax.patches:
    height = p.get_height()
    if height > 0.005:  
        ax.annotate(f"{height:.2%}",
                    (p.get_x() + p.get_width() / 2, height),
                    ha='center', va='bottom',
                    fontsize=15)
    
plt.show()

#%% 
# Repayment Statuses against Default Status 
# Repayment Status vs. Default Visualizations 
# columns
pay_status_cols =  [
    "April_Pay_status", "May_Pay_status", "June_Pay_status",
    "July_Pay_status", "August_Pay_status", "Sept_Pay_status"
]

# count plots for each column
# figure size
# (width, height)
plt.figure(figsize=(30, 12))

# columns
for i, col in enumerate(pay_status_cols, 1):
    plt.subplot(2, 4, i)
    ax = sns.countplot(x=col, hue='default_payment_next_month', data=clients_data, palette='Set2')
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
               fontsize = 18)

plt.show()

#%%
# Repayment Status (September Only) agaginst Default
# Figure Size
plt.figure(figsize=(12, 10))

# visualization itself
sns.countplot(x="Sept_Pay_status", 
              hue='default_payment_next_month', 
              data=clients_data, 
              palette='Set2')

# Design 
# title
plt.title(f'September Payments Distribution by Default Status', fontweight = "bold", fontsize = 24)

# xlabel
plt.xlabel(f'September Pay Status', fontsize = 20)

# ylabel
plt.ylabel('Count', fontsize = 20)
plt.tick_params(axis='both', labelsize=18)

# legend
plt.legend(title = "Default Status",
            title_fontsize = 20,
            fontsize = 18)

plt.show()
# %%
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

#%%
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
          fontdict= {"fontsize":24, 
                     "fontweight":"bold"})
plt.xlabel("Credit Limit", 
           fontdict= {"fontsize": 20})

plt.ylabel("Default Status", 
           fontdict= {"fontsize": 20})

plt.tick_params(axis='both', labelsize=18)

plt.show()
#%%
# Credit Limit vs Default Status by Sex
plt.figure(figsize= (12, 10))
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

# Legend 
plt.legend("Female", 
           "Male")

# show the plot
plt.show()

# Credit Limit vs Default Status by Marriage Status
plt.figure(figsize= (12, 10))
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

# Legend 
plt.legend(["Married", 
            "Single"])
plt.show()

# %%
# proportions
# figure size
plt.figure(figsize=(30, 20))

# columns
for i, col in enumerate(pay_status_cols, 1):
    plt.subplot(2, 4, i)
    ax = sns.histplot(x=col, hue='default_payment_next_month', data=clients_data, multiple='fill', palette='Set2')
  
    # xlabel
    plt.xlabel(f'{col} Status', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
    
    # ylabel
    plt.ylabel('Proportion', fontsize=14)  
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)

    # legend
    plt.legend(title="Default Status", title_fontsize=18, fontsize=14)

plt.show()

#%%[markdown]
## Statistical Testing 
# Statistical tests

# For Numeric Variables

# Recreate Numeric Columns 
numeric_cols = clients_data.select_dtypes(include='number')
numeric_cols = numeric_cols.drop(columns=['default_status'], errors='ignore')


# 1. t-tests / Mann-Whitney (non-parametric) U Test for Numerical Features
print("\n### T-Tests / Mann-Whitney U Tests for Numerical Features ###\n")

for col in numeric_cols.columns:
    group0 = clients_data[clients_data['default_payment_next_month'] == 0][col]
    group1 = clients_data[clients_data['default_payment_next_month'] == 1][col]

    # First check if both groups are normally distributed
    stat0, p0 = shapiro(group0)
    stat1, p1 = shapiro(group1)

    if p0 > 0.05 and p1 > 0.05:
        # If both are normal, use t-test
        stat, p = ttest_ind(group0, group1)
        test_name = "T-test"
    else:
        # Otherwise, use Mann-Whitney U Test
        stat, p = mannwhitneyu(group0, group1, alternative='two-sided')
        test_name = "Mann-Whitney U"

    print(f"{col}: {test_name}, statistic = {stat:.4f}, p-value = {p:.4f}")
    # Interpretation: p < 0.05 means significant difference between default groups
    # But tests are just for screening, do not drop predictors based on that.

#%%
# 2. Categorical Chi-Square test of Independence 
# Catgorical Variables
# Make all required categorical transformations
# Convert selected columns to categorical
categorical_columns = [
    'MARRIAGE', 
    'SEX', 
    'EDUCATION'
]


clients_data_converted = clients_data.copy()

# Convert to category dtype
for col in categorical_columns:
    clients_data_converted[col] = clients_data_converted[col].astype('category')

# Chi-square Tests for Categorical Features
print("\n### Chi-Square Tests for Categorical Features ###\n")

for col in categorical_columns:
    contingency_table = pd.crosstab(clients_data_converted[col], 
                                    clients_data_converted['default_payment_next_month'])
    stat, p, dof, expected = chi2_contingency(contingency_table)
    print(f"{col}: Chi-square statistic = {stat:.4f}, p-value = {p:.4f}")
    # Interpretation: p < 0.05 means feature and default status are dependent


#%%
# Correlation Matrix
full_corr_matrix = numeric_cols.corr()

# Plot heatmap
plt.figure(figsize=(21, 15))
sns.heatmap(
    full_corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)

# Design
# Title
plt.title('Correlation Matrix: Correlated Features with Default Status', fontsize=18, fontweight='bold')

# X-axis
plt.xticks(rotation=45, ha='right')

# Y-axis
plt.yticks(rotation=0)

plt.show()

#%% 
# Correlation against dependent (default status)

# Check correlations
# Select only numeric columns
numeric_cols = clients_data_converted.select_dtypes(include='number')

# Pearson correlation
pearson_corr = numeric_cols.corr()['default_payment_next_month']

# Sort and plot
correlations_with_target = pearson_corr.reindex(pearson_corr.abs().sort_values(ascending=False).index)

# Check correlations again
numeric_cols = clients_data_converted.select_dtypes(include='number')

# Pearson correlation
pearson_corr = numeric_cols.corr()['default_payment_next_month'].drop('default_payment_next_month')

# Print
print(f"Correlations with Default Status:\n{correlations_with_target}\n")

# Plot
plt.figure(figsize=(12, 10))
pearson_corr.plot(kind='barh')

# Invert y-axis
plt.gca().invert_yaxis()

# Design
# Titlee
plt.title('Correlations with Default Status (Highest to Lowest)', fontsize=18, fontweight='bold')

# X-axis
plt.xlabel('Correlation Coefficient', fontsize=14)
plt.xticks(fontsize=12)

# Y-axis
plt.ylabel('Features', fontsize=14)
plt.yticks(fontsize=12)

plt.show()
#%%[markdown]
## Initial Modeling
# Manipulating the Education Ordinal Relationship 

# Create ordinal relationship
# 0 (graduate) - lowest risk
# 1 (undergraduate) - increasing risk
# 2 (high scool) - highest risk
clients_data_converted["EDUCATION"] = clients_data_converted["EDUCATION"].map({1: 0, 2: 1, 3: 2})

# Check again
clients_data_converted["EDUCATION"].value_counts(normalize= True)

# %% 
# Polynomial Testing
# We want to see if by converting some of the numerical variables to Polynomials/Log, there is a gain in F1 metric
# Let's test polynomials need


# Selected continuous variables for transformation
continuous_candidates = [
    'LIMIT_BAL',
    'AGE',
    'Sept_Bill_Amount',
      'August_Bill_Amount', 'July_Bill_Amount', 'June_Bill_Amount',
      'May_Bill_Amount', 'April_Bill_Amount', 'Sept_Pay_Amount',
      'August_Pay_Amount', 'July_Pay_Amount', 'June_Pay_Amount',
      'May_Pay_Amount', 'April_Pay_Amount'
]

results = []

# Target variable
y = clients_data_converted['default_payment_next_month']

# Loop through each continuous variable and test polynomial and log transforms
for col in continuous_candidates:
    df_temp = clients_data_converted.copy()

    # Create transformed versions
    df_temp[f'{col}_squared'] = df_temp[col] ** 2
    df_temp[f'{col}_log'] = np.log1p(df_temp[col].clip(lower=0))

    # Prepare datasets
    base_X = df_temp[[col]]
    squared_X = df_temp[[col, f'{col}_squared']]
    log_X = df_temp[[col, f'{col}_log']]

    model = LogisticRegression(max_iter=1000)

    # Cross-validated F1 scores
    f1_base = cross_val_score(model, base_X, y, scoring='f1', cv=5).mean()
    f1_sq = cross_val_score(model, squared_X, y, scoring='f1', cv=5).mean()
    f1_log = cross_val_score(model, log_X, y, scoring='f1', cv=5).mean()

    results.append({
        'feature': col,
        'f1_base': f1_base,
        'f1_with_squared': f1_sq,
        'f1_with_log': f1_log,
        'best_transformation': max(
            [(f1_base, 'base'), (f1_sq, 'squared'), (f1_log, 'log')],
            key=lambda x: x[0]
        )[1]
    })

results_df = pd.DataFrame(results).sort_values(by='f1_with_squared', ascending=False)
results_df
# There is a need of feature engineering 

#%%
# First, baseline results 
#%%
# Using Pycaret Library 
# Setup the environment
setup(
    # dataset name
    data = clients_data_converted,

    # dependent variable
    target='default_payment_next_month',

    # session id for consistency and reproducibility
    session_id=42,

    # folds for cross-validation
    fold_strategy='stratifiedkfold',
    fold=5,

    # parallelization
    n_jobs = -1,

    # experiment name
    experiment_name='baseline')

#%% 
# Give baseline models
compare_models(sort = "F1")

# %%
# Crete A Momentum Variable 
clients_data_converted["Momentum"] = clients_data_converted.apply(calculate_weighted_momentum, axis=1)

# dynamic column search for repayment status
rp_columns = [col for col in clients_data_converted.columns if 'pay' in col.lower() and 'status' in col.lower()]

# average repayment status
clients_data_converted["average_repayment_status"] = clients_data_converted[rp_columns].mean(axis=1)

# repayment volatility
clients_data_converted['repayment_volatility'] = clients_data_converted[rp_columns].std(axis = 1)

# Repayment in recent months
recent_months = ["July_Pay_status", "August_Pay_status", "Sept_Pay_status"]
clients_data_converted["recent_repayment_mean"] = clients_data_converted[recent_months].mean(axis=1)

# Momentum and Volatility
clients_data_converted["momentum_volatility_interaction"] = clients_data_converted["Momentum"] * clients_data_converted["repayment_volatility"]

# Number of Low Repayment Status months
clients_data_converted["low_repayment_months"] = (clients_data_converted[rp_columns] > 1).sum(axis=1)

# Risk Index
clients_data_converted["risk_index_1"] = clients_data_converted["momentum_volatility_interaction"] + 0.5 * clients_data_converted["low_repayment_months"]

# Repayment deterioration and acceleration
clients_data_converted["repayment_deterioration"] = clients_data_converted["Sept_Pay_status"] - clients_data_converted["June_Pay_status"]

clients_data_converted["repayment_acceleration"] = (
    (clients_data_converted["Sept_Pay_status"] - clients_data_converted["August_Pay_status"]) -
    (clients_data_converted["August_Pay_status"] - clients_data_converted["July_Pay_status"])
)

# Momentum Recent Mean Interaction
clients_data_converted['momentum_recent_mean_interaction'] = clients_data_converted['Momentum'] * clients_data_converted['recent_repayment_mean']

# Momentum Stability Flag
clients_data_converted['momentum_stability_flag'] = np.where(
    (clients_data_converted['Momentum'] == 0) & (clients_data_converted['repayment_volatility'] < 1.0), 1, 0
).astype('int64')

# Final Formula
clients_data_converted["super_default_score_final"] = (
    0.25 * clients_data_converted["low_repayment_months"] +                # Missed payments (strongest)
    0.20 * clients_data_converted["recent_repayment_mean"] +               # Recency behavior
    0.10 * clients_data_converted["momentum_recent_mean_interaction"] +    # Recency + momentum
    0.10 * clients_data_converted["Momentum"] +                             # Worsening trend
    0.10 * clients_data_converted["momentum_volatility_interaction"] +     # Instability + trend
    0.10 * clients_data_converted["average_repayment_status"] +            # Long-term repayment status
    0.10 * clients_data_converted["risk_index_1"] -                         # Risk score from multiple signals
    0.10 * clients_data_converted["momentum_stability_flag"]               # Penalize stable-good behavior
)

# Average features 
# bill amounts and payment amounts

# bill columns
bill_columns = [col for col in clients_data_converted.columns if 'bill' in col.lower() and 'amount' in col.lower()]

# payment columns
payment_columns = [col for col in clients_data_converted.columns if 'pay' in col.lower() and 'amount' in col.lower() and 'status' not in col.lower()]

# Calculate the average of bills and payments
clients_data_converted['average_bill'] = clients_data_converted[bill_columns].mean(axis=1)
clients_data_converted['average_payment'] = clients_data_converted[payment_columns].mean(axis=1)

# check new data
clients_data_converted.info()

#%% 
# Correlation against dependent (default status)

# Check correlations
# Select only numeric columns
numeric_cols = clients_data_converted.select_dtypes(include='number')

# Pearson correlation
pearson_corr = numeric_cols.corr()['default_payment_next_month']

# Sort and plot
correlations_with_target = pearson_corr.reindex(pearson_corr.abs().sort_values(ascending=False).index)

# Check correlations again
numeric_cols = clients_data_converted.select_dtypes(include='number')

# Pearson correlation
pearson_corr = numeric_cols.corr()['default_payment_next_month'].drop('default_payment_next_month')

# Print
print(f"Correlations with Default Status:\n{correlations_with_target}\n")

# Plot
plt.figure(figsize=(12, 10))
pearson_corr.plot(kind='barh')

# Invert y-axis
plt.gca().invert_yaxis()

# Design
# Titlee
plt.title('Correlations with Default Status (Highest to Lowest)', fontsize=18, fontweight='bold')

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

# Plot heatmap
plt.figure(figsize=(30, 20))
sns.heatmap(
    full_corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)

# Design
# Title
plt.title('Correlation Matrix:  Correlated Features with Default Status', fontsize=18, fontweight='bold')

# X-axis
plt.xticks(rotation=45, ha='right')

# Y-axis
plt.yticks(rotation=0)

plt.show()

#%%
# Now Let's see if Some variables benefit from logging again 

# Selected continuous variables for transformation
continuous_candidates = [
    'LIMIT_BAL',
    'AGE',
    'Sept_Bill_Amount',
      'August_Bill_Amount', 'July_Bill_Amount', 'June_Bill_Amount',
      'May_Bill_Amount', 'April_Bill_Amount', 'Sept_Pay_Amount',
      'August_Pay_Amount', 'July_Pay_Amount', 'June_Pay_Amount',
      'May_Pay_Amount', 'April_Pay_Amount',
    'repayment_volatility',
    'momentum_volatility_interaction',
    'risk_index_1',
    'super_default_score_final',
    'recent_repayment_mean',
    'average_bill',
    'average_payment',
    'repayment_deterioration',
    'repayment_acceleration',
    'momentum_recent_mean_interaction',
    'momentum_stability_flag',
    'low_repayment_months',
    'average_repayment_status'
]

results = []

# Drop NaNs from ratio column
df_test = clients_data_converted.copy()

# Target variable
y = df_test['default_payment_next_month']

# Loop through each continuous variable and test polynomial and log transforms
for col in continuous_candidates:
    df_temp = df_test.copy()

    # Create transformed versions
    df_temp[f'{col}_squared'] = df_temp[col] ** 2
    df_temp[f'{col}_log'] = np.log1p(df_temp[col].clip(lower=0))

    # Prepare datasets
    base_X = df_temp[[col]]
    squared_X = df_temp[[col, f'{col}_squared']]
    log_X = df_temp[[col, f'{col}_log']]

    model = LogisticRegression(max_iter=1000)

    # Cross-validated F1 scores
    f1_base = cross_val_score(model, base_X, y, scoring='f1', cv=5).mean()
    f1_sq = cross_val_score(model, squared_X, y, scoring='f1', cv=5).mean()
    f1_log = cross_val_score(model, log_X, y, scoring='f1', cv=5).mean()

    results.append({
        'feature': col,
        'f1_base': f1_base,
        'f1_with_squared': f1_sq,
        'f1_with_log': f1_log,
        'best_transformation': max(
            [(f1_base, 'base'), (f1_sq, 'squared'), (f1_log, 'log')],
            key=lambda x: x[0]
        )[1]
    })

results_df = pd.DataFrame(results).sort_values(by='f1_with_squared', ascending=False)
results_df
# %%
# Select features where log transformation had best F1 and improved over base
final_features = [
    "super_default_score_final",
    "risk_index_1",
    "low_repayment_months",
    "momentum_recent_mean_interaction",
    "recent_repayment_mean",
    "average_repayment_status",
    "momentum_volatility_interaction"
]


# Create log-transformed versions of those features
clients_data_logs = clients_data_converted.copy()

for col in final_features:
    clients_data_logs[f'{col}_log'] = np.log1p(clients_data_logs[col].clip(lower=0))
    clients_data_logs.drop(columns=f'{col}', inplace=True)

# info
clients_data_logs.info()

#%%
# filter and include variables with corelation higher than 0.15 in ne dataframe
# Calculate correlation with the target
correlation_matrix = clients_data_logs.corr()
target_corr = correlation_matrix["default_payment_next_month"]

# Filter features with absolute correlation > 0.15 (excluding the target itself)
selected_features = target_corr[abs(target_corr) > 0.15].index.tolist()
selected_features = [col for col in selected_features if col != "default_payment_next_month"]

# Create new DataFrame with selected features + target
clients_data_filtered = clients_data_logs[selected_features + ["default_payment_next_month"]]

# View selected feature correlations
print("Selected Features with correlation > 0.15:")
print(target_corr[abs(target_corr) > 0.15])

## Feature Selection 
# Using Forward Feature Selection and Efficiency 
# Forward Feature Selection found the most combintion
#%%


# Variables
# Target variable
Y = clients_data_filtered["default_payment_next_month"]

# Feature candidates
X = clients_data_filtered.drop(columns=["default_payment_next_month"])


# Light Gradient Boosting Classifier as selector,as it is: 
# Redundant to noise, can find non-linear relationships, doesn't care about multicolliearity
# Pretty fast and efficient 

# Define model
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    class_weight="balanced"
)


# Build pipeline: scaling + model
pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('clf', model)
])

# Setup Sequential Feature Selector
sfs = SFS(estimator=pipeline,
          k_features="parsimonious",
          forward=True,
          floating=True,
          scoring=make_scorer(f1_score),
          cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
          n_jobs=-1,
          verbose=2)

# Run SFS
start_time = time.time()
sfs.fit(X, Y)
end_time = time.time()

# Extract Selected Features
selected_features = list(sfs.k_feature_names_)
print(f"\n Selected Features ({len(selected_features)}):\n", selected_features)
print(f" Feature selection completed in {end_time - start_time:.2f} seconds")

# %%
# Best variables combination is: 
# variables = ['Sept_Pay_status', 
#             'August_Pay_status', 
#             'July_Pay_status',
#             'June_Pay_status', 
#             'May_Pay_status',
#         'Momentum',
#         'momentum_stability_flag',
#         'risk_index_1_log',
#         'low_repayment_months_log',
#         'momentum_recent_mean_interaction_log',
#         'recent_repayment_mean_log',
#         'average_repayment_status_log',
#             'momentum_volatility_interaction_log']

from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold

# Best variable combination is: 
variables = ['Sept_Pay_status', 
             'June_Pay_status', 
             'May_Pay_status', 
             'momentum_stability_flag',
            'low_repayment_months_log', 
            'momentum_recent_mean_interaction_log']


#%%
# Initial Modeling
# df, columns, target="default_payment_next_month", plot_name=None, max_iter=1000, test_size=0.2
log_reg_model = run_logistic_regression(
    clients_data_filtered,
    variables,
    # target="default_payment_next_month",
    plot_name="Initial Logistic Regression",
)

rand_forest_model = run_random_forest_classifier(
    clients_data_filtered,
    variables,
    # target="default_payment_next_month",
    # plot_name="Initial Random Forest Classifier",
)


#%%

# Let's use it in cross-validation 
# We will use STOMETomek oversampling, RobustScaler for normalizations, and 
# StratifiedCrossValidation with 5 folds for combatting the class imbalance. 

# Features and target
Y = clients_data_filtered["default_payment_next_month"]
X =  clients_data_filtered[variables]

# Build the pipeline
pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('smote_tomek', SMOTETomek(random_state=42)),
    ('clf', LGBMClassifier())
])

# stratified Cross-Vaidation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate
scores = cross_val_score(pipeline, X, Y, scoring='f1', cv=skf)
print("Stratified F1 scores:", scores)
print("Mean Stratified F1 score:", scores.mean())
# Average F1 across 5 folds 
# With this combination is: 0.5322% 

#%%
# Using Backward Selection 
# Variables
# Target variable
Y = clients_data_filtered["default_payment_next_month"]

# Feature candidates
X = clients_data_filtered.drop(columns=["default_payment_next_month"])


# Light Gradient Boosting Classifier as selector,as it is: 
# Redundant to noise, can find non-linear relationships, doesn't care about multicolliearity
# Pretty fast and efficient 

# Define model
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    class_weight="balanced"
)


# Build pipeline: scaling + model
pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('clf', model)
])

# Setup Sequential Feature Selector
sfs = SFS(estimator=pipeline,
          k_features="parsimonious",
          forward=False,
          floating=True,
          scoring=make_scorer(f1_score),
          cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
          n_jobs=-1,
          verbose=2)

# Run SFS
start_time = time.time()
sfs.fit(X, Y)
end_time = time.time()

# Extract Selected Features
selected_features = list(sfs.k_feature_names_)
print(f"\n Selected Features ({len(selected_features)}):\n", selected_features)
print(f" Feature selection completed in {end_time - start_time:.2f} seconds")

#%%
# Best Variables combination check for Backward Selection
# Best variable combination is: 
variables = ['Sept_Pay_status', 
             'May_Pay_status', 
             'momentum_stability_flag', 
             'low_repayment_months_log']

# Let's use it in cross-validation and test
# We will use STOMETomek oversampling, RobustScaler for normalizations, and 
# StratifiedCrossValidation with 5 folds for combatting the class imbalance. 

# Features and target
Y = clients_data_filtered["default_payment_next_month"]
X =  clients_data_filtered[variables]

# Build the pipeline
pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('smote_tomek', SMOTETomek(random_state=42)),
    ('clf', LGBMClassifier())
])

# stratified Cross-Vaidation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate
scores = cross_val_score(pipeline, X, Y, scoring='f1', cv=skf)
print("Stratified F1 scores:", scores)
print("Mean Stratified F1 score:", scores.mean())
# Average F1 across 5 folds 
# With this combination is: 0.5342%
#%% [markdown]
## Final Modeling + Tuning
# Let's use  our variables combination we found and fit them to Pycaret for all models 
# Backward Selection PyCaret
# variables
variables = ['Sept_Pay_status', 
             'May_Pay_status', 
             'momentum_stability_flag', 
             'low_repayment_months_log']

# Create a new dataframe with selected features and target
df_pycaret = clients_data_filtered[variables + ['default_payment_next_month']]

# Set up PyCaret classification environment
clf_setup = setup(
    # data
    data=df_pycaret,

    # target
    target='default_payment_next_month',

    # reproducibility
    session_id=42,

    # normalization
    normalize=True,
    normalize_method= "robust",

    # imbalance
    fix_imbalance=True,
    fix_imbalance_method= "smotetomek",

    # folds
    # folds for cross-validation
    fold_strategy='stratifiedkfold',
    fold=5,

    # experimentation
    experiment_name = "baseline_after_selection",

    verbose=True)

# Compare all models
best_models = compare_models(sort='F1')
# %% [markdown]
#### Gradient Boosting Classifier
# create the model
gbc_model = create_model("gbc")

# predict the model on test data
predict_model(gbc_model)

#%%
# tune model
tuned_gbc_model = tune_model(gbc_model,
                                 optimize = "f1",
                                  n_iter = 50,
                                 choose_better= True)

# check the formula 
tuned_gbc_model

#%%
# write down manually incase if session crashes
tuned_gbc_model = create_model("gbc",
                            ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.4, loss='log_loss', max_depth=1,
                           max_features=1.0, max_leaf_nodes=None,
                           min_impurity_decrease=0.4, min_samples_leaf=4,
                           min_samples_split=2, min_weight_fraction_leaf=0.0,
                           n_estimators=130, n_iter_no_change=None,
                           random_state=42, subsample=0.6, tol=0.0001,
                           validation_fraction=0.1, verbose=True,
                           warm_start=False)
# predict the tuned model on testing data
predict_model(tuned_gbc_model)
#%%
# plots for the model 
# auc
plot_model(tuned_gbc_model, plot = 'auc')

# confusion matrix
plot_model(tuned_gbc_model, plot = 'confusion_matrix', plot_kwargs= {"percent": True})

# precision-recall curve
plot_model(tuned_gbc_model, plot = 'pr')

# classification report
plot_model(tuned_gbc_model, plot = 'class_report')

# lift curve
plot_model(tuned_gbc_model, plot = 'lift')

# gain
plot_model(tuned_gbc_model, plot = 'gain')

# feature importance
plot_model(tuned_gbc_model, plot = 'feature')

#%% [markdown]
#### Ada Boosting Classifier
# create ada model
ada_model = create_model('ada')

# predict on test data
predict_model(ada_model)

#%%
# tune model 
tuned_ada_model = tune_model(ada_model,
                                 optimize = "f1",
                                  n_iter = 50,
                                 choose_better= True)

# check the formula
tuned_ada_model

#%%
# type manually
tuned_ada_model = create_model("ada", algorithm='SAMME', learning_rate=0.1,
                   n_estimators=180, random_state=42)
# predict tuned model on test data
predict_model(tuned_ada_model)

#%%
#%% plots for the model 
# AUC
plot_model(tuned_ada_model, plot = 'auc')

# confusion matrix
plot_model(tuned_ada_model, plot = 'confusion_matrix', plot_kwargs= {"percent": True})

# precision-recall curve
plot_model(tuned_ada_model, plot = 'pr')

# classification report
plot_model(tuned_ada_model, plot = 'class_report')

# lift curve
plot_model(tuned_ada_model, plot = 'lift')

# gain
plot_model(tuned_ada_model, plot = 'gain')

# feature importance
plot_model(tuned_ada_model, plot = 'feature')

#%%[markdown]
#### Light-Gradient Boosting Model 
# create the model
lgbm_model = create_model('lightgbm')
                          
# predict on test data
predict_model(lgbm_model)

#%%
# tune model
tuned_lgbm = tune_model(lgbm_model,
                        optimize = "f1",
                                  n_iter = 50,
                                 choose_better= True)

# Check the formula
tuned_lgbm

#%%
# Create model manually in case if session crashes
tuned_lgbm = create_model('lightgbm',
                          bagging_fraction=0.6, bagging_freq=5, boosting_type='gbdt',
               class_weight=None, colsample_bytree=1.0, feature_fraction=0.8,
               importance_type='split', learning_rate=0.05, max_depth=-1,
               min_child_samples=86, min_child_weight=0.001, min_split_gain=0.9,
               n_estimators=130, n_jobs=-1, num_leaves=40, objective=None,
               random_state=42, reg_alpha=2, reg_lambda=0.001, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)

# predict on test data
predict_model(tuned_lgbm)

#%% 
# roc-auc
plot_model(tuned_lgbm, plot = 'auc')

# confusion matrix
plot_model(tuned_lgbm, plot = 'confusion_matrix', plot_kwargs= {"percent": True})
plot_model(tuned_lgbm, plot = 'confusion_matrix')

# precision-recall curve
plot_model(tuned_lgbm, plot = 'pr')

# classification report
plot_model(tuned_lgbm, plot = 'class_report')

# lift curve
plot_model(tuned_lgbm, plot = 'lift')

# gain
plot_model(tuned_lgbm, plot = 'gain')

# feature importance
plot_model(tuned_lgbm, plot = 'feature')


# %%
#### Logistic Regression
# create the model
logistic_model = create_model('lr')
                          
# predict on test data
predict_model(logistic_model)

#%%
# tune model
tuned_logistic = tune_model(logistic_model,
                        optimize = "f1",
                                  n_iter = 50,
                                 choose_better= True)

# Check the formula
tuned_logistic
# Logistic Function failed to tune and improve results
# Use casual logstic function
#%%
# predict on test data
predict_model(logistic_model)

#%%
# We try optimizing our logistic regression even more 
# by threshold optimization, finding prob.threshold that is better than 0.5
optimized_logistic = optimize_threshold(logistic_model, 
                                        optimize = "f1", 
                                        return_data = True)
# threshold optimization failed - 0.5 the best threshold for us 
# We still hav few options, let's try to create a logistic ensemble model

# %%
# Ensemble Model out of logistic regression
# bagging method 
bagged_logistic = ensemble_model(logistic_model, 
                                 optimize= "f1", 
                                choose_better = True)
# bagging also failed, as original was better.
#%%
# boosting method 
boost_logistic = ensemble_model(logistic_model, 
                                method = "Boosting",
                                optimize= "f1", 
                                choose_better = True)
# boosting also failed, as original model was better. 
# This is the maximum we can achieve with logistic model. 
#%% 
# roc-auc
plot_model(tuned_logistic, plot = 'auc')

# confusion matrix
plot_model(tuned_logistic, plot = 'confusion_matrix', plot_kwargs= {"percent": True})

# precision-recall curve
plot_model(tuned_logistic, plot = 'pr')

# classification report
plot_model(tuned_logistic, plot = 'class_report')

# lift curve
plot_model(tuned_logistic, plot = 'lift')

# gain
plot_model(tuned_logistic, plot = 'gain')

# feature importance
plot_model(tuned_logistic, plot = 'feature')
#%%
### Experimental:
### Blending
# Blending and Stacking Models that we have
# Let's do Blending First
blender = blend_models([tuned_gbc_model, 
                       tuned_ada_model, 
                       tuned_lgbm, 
                       logistic_model],
                       optimize= "f1", 
                       choose_better= True)
# Blending models failed, original model was returned
# Let's remove weakeest model and try again
#%%
blender_trees = blend_models([tuned_gbc_model, 
                       tuned_ada_model, 
                       tuned_lgbm],
                       optimize= "f1", 
                       choose_better= True)
# Failed again, let's remove weakest model again
#%%
blender_top = blend_models([tuned_gbc_model, 
                       tuned_lgbm],
                       optimize= "f1", 
                       choose_better= True)
# Still worse than single model, we can't do anything more at blending
# Let's try stacking. 
#%%
### Stacking
# First, all models 
stacker = stack_models([tuned_gbc_model, 
                       tuned_ada_model, 
                       tuned_lgbm, 
                       logistic_model],
                       optimize= "f1", 
                       choose_better= True, 
                       # choosing meta model the best that we have 
                       meta_model= lgbm_model, 
                       restack= True)
# Worse than a single tuned lightGBM model, however let's again remove 
# weakest model. 
#%%
stacker_trees = stack_models([tuned_gbc_model, 
                       tuned_ada_model, 
                       tuned_lgbm],
                       optimize= "f1", 
                       choose_better= True, 
                       # choosing meta model the best that we have 
                       meta_model= lgbm_model, 
                       restack= True)
# Again, it's not working. 
# Let's remove last weakest model and see how it's going to react. 
#%%
stacker_top = stack_models([tuned_gbc_model, 
                       tuned_lgbm],
                       optimize= "f1", 
                       choose_better= True, 
                       # choosing meta model the best that we have 
                       meta_model= lgbm_model, 
                       restack= True)
# Neither stacking or blending work. So our best model is - LIGHTGBM
