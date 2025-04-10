#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, precision_score, f1_score, recall_score, roc_curve, auc, ConfusionMatrixDisplay


#%%
# Functions
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

def run_linear_regression(df, columns, target="default_payment_next_month", plot=False, max_iter=1000, test_size=0.2):
    X = df[columns]
    y = df[target]

    model = LogisticRegression(max_iter=max_iter)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    threshold = 0.4  # Adjust as needed
    y_pred = (y_pred_proba >= threshold).astype(int)
    mse = mean_squared_error(y_test, y_pred)
    
    rmse = np.sqrt(mse)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    if plot == True:
        plot_logistic_regression_results(y_test, y_pred, y_pred_proba)
    
    return rmse, accuracy, precision, conf_matrix, recall, f1

def print_model_results(rmse, accuracy, precision, conf_matrix, recall, f1, model_number):
    print(f"RMSE {model_number}: {rmse}")
    print(f"Accuracy {model_number}: {accuracy}")
    print(f"Precision {model_number}: {precision}")
    print(f"Confusion Matrix {model_number}:\n{conf_matrix}")
    print(f"Recall {model_number}: {recall}")
    print(f"F1 Score {model_number}: {f1}")
    print()
    
def plot_logistic_regression_results(y_test, y_pred, y_pred_proba, model_name="Logistic Regression"):
    # Plot Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()

    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

#%%
# Load data
# clients_data = load_data("../data/credit_card_clients_full.csv")
clients_data = pd.read_csv("../data/credit_card_clients_full.csv")

# Drop unnecessary columns
clients_data.drop(columns=["Unnamed: 0"], inplace=True, axis=1)

#%%
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

# Calculate momentum based on payment status
# Define the payment status columns in chronological order
pay_status_cols = [
    "April_Pay_status", "May_Pay_status", "June_Pay_status",
    "July_Pay_status", "August_Pay_status", "Sept_Pay_status"
]



clients_data["Momentum"] = clients_data.apply(calculate_weighted_momentum, axis=1)
clients_data["Momentum_Label"] = clients_data["Momentum"].map({0: "Stable/Improving", 1: "Bad Momentum"})

#%%
# View the resulting dataframe
clients_data.head()

#%%
plot_correlation(clients_data, feature_cols=['Momentum', 'Sept_Pay_status', 'LIMIT_BAL'], method="spearman", target_col = "default_payment_next_month")

#%%
clients_encoded = pd.get_dummies(clients_data, columns=['Momentum','Sept_Pay_status'], drop_first=True)

#%%[markdown]
# None of the features are too closely correlated with each other, although corrlated moderately
# 
# We will try multiple models including all of them
# 
# 
## Logistic Regression Model
# 



# %%
# Model 1: Sept Pay status, Momentum, LIMIT_BAL
model_1_results = run_linear_regression(clients_encoded, ['Momentum_1', 'Sept_Pay_status_-1',
       'Sept_Pay_status_0', 'Sept_Pay_status_1', 'Sept_Pay_status_2',
       'Sept_Pay_status_3', 'Sept_Pay_status_4', 'Sept_Pay_status_5',
       'Sept_Pay_status_6', 'Sept_Pay_status_7', 'Sept_Pay_status_8',
       'LIMIT_BAL'], plot = False)
# print_model_results(*model_1_results, "model 1")
print(f"Model 1 F1: {model_1_results[-1]}")

# Model 2: Momentum, LIMIT_BAL
# rmse_model_2, accuracy_model_2, precision_model_2, conf_matrix_model_2, recall_model_2, f1_model_2 = run_linear_regression(clients_encoded, ['Momentum_1', 'LIMIT_BAL'])
model_2_results = run_linear_regression(clients_encoded, ['Momentum_1', 'LIMIT_BAL'])
# print_model_results(*model_2_results, "model 2")
print(f"Model 2 F1: {model_2_results[-1]}")


# Model 3: Sept Pay status, LIMIT_BAL
model_3_results = run_linear_regression(clients_encoded, ['Sept_Pay_status_-1',
       'Sept_Pay_status_0', 'Sept_Pay_status_1', 'Sept_Pay_status_2',
       'Sept_Pay_status_3', 'Sept_Pay_status_4', 'Sept_Pay_status_5',
       'Sept_Pay_status_6', 'Sept_Pay_status_7', 'Sept_Pay_status_8','LIMIT_BAL'])
# print_model_results(*model_3_results, "model 3")
print(f"Model 3 F1: {model_3_results[-1]}")


# Model 4: LIMIT_BAL
model_4_results = run_linear_regression(clients_encoded, ['LIMIT_BAL'])
# print_model_results(*model_4_results, "model 4")
print(f"Model 4 F1: {model_4_results[-1]}")

# Model 5: Sept Pay status
model_5_results = run_linear_regression(clients_encoded, ['Sept_Pay_status_-1',
       'Sept_Pay_status_0', 'Sept_Pay_status_1', 'Sept_Pay_status_2',
       'Sept_Pay_status_3', 'Sept_Pay_status_4', 'Sept_Pay_status_5',
       'Sept_Pay_status_6', 'Sept_Pay_status_7', 'Sept_Pay_status_8'])
# print_model_results(*model_5_results, "model 5")
print(f"Model 5 F1: {model_5_results[-1]}")
print()



# %%[markdown]
## Analysis of Results
# Should ignore accuracy for now, as the dataset is unbalanced
# 
# For models 1, 2, and 3, the precision is 

#%%
# I want the number of Momentum labels == "Stable/Improving" and "Bad Momentum"
num_1 = clients_data[clients_data["Momentum_Label"] == "Stable/Improving"].shape[0]
num_0 = clients_data[clients_data["Momentum_Label"] == "Bad Momentum"].shape[0]


print(f"Number of Stable/Improving: {num_1}")
print(f"Number of Bad Momentum: {num_0}")

# %%
# def plot_logistic_regression_results(y_test, y_pred, y_pred_proba, model_name="Logistic Regression"):
# rmse, accuracy, precision, conf_matrix, recall, f1
# plot_logistic_regression_results(model_1_results[3], model_1_results[4], model_1_results[2], model_name="Logistic Regression Model 1")
# plot_logistic_regression_results(model_2_results[3], model_2_results[4], model_2_results[2], model_name="Logistic Regression Model 2")
# %%
