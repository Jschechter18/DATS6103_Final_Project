
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

#%%
import pandas as pd
import numpy as np

# Load the original dataset
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

# Fix categorical inconsistencies
df['Education_Level'] = df['Education_Level'].replace({0: 4, 5: 4, 6: 4})
df['Marital_Status'] = df['Marital_Status'].replace({0: 3})

# Create Momentum
def calculate_weighted_momentum(row):
    cols = ['Repayment_Status_Sep', 'Repayment_Status_Aug', 'Repayment_Status_Jul',
            'Repayment_Status_Jun', 'Repayment_Status_May', 'Repayment_Status_Apr']
    weights = [6, 5, 4, 3, 2, 1]
    return "Bad Momentum" if sum(weights[i] * row[cols[i]] for i in range(6)) > 20 else "Stable/Improving"
df['Momentum'] = df.apply(calculate_weighted_momentum, axis=1)

# Create Payment-to-Bill Ratio
bill_cols = [f'Bill_Amount_{m}' for m in ['Sep', 'Aug', 'Jul', 'Jun', 'May', 'Apr']]
pay_cols = [f'Payment_Amount_{m}' for m in ['Sep', 'Aug', 'Jul', 'Jun', 'May', 'Apr']]
df['Payment_Bill_Ratio'] = df[pay_cols].mean(axis=1) / df[bill_cols].mean(axis=1)
df['Payment_Bill_Ratio'] = df['Payment_Bill_Ratio'].replace([np.inf, -np.inf], 0).fillna(0)

# Save to cleaned CSV
df.to_csv("credit_card_clients_cleaned.csv", index=False)
print("Dataset successfully saved as credit_card_clients_cleaned.csv")


#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap
import joblib
from sklearn.metrics import RocCurveDisplay


# Load cleaned dataset
df = pd.read_csv("credit_card_clients_cleaned.csv")
 
# Drop unnecessary columns and encode target
X = df.drop(columns=['Client_ID', 'Default_Payment'])
X = pd.get_dummies(X, drop_first=True)
y = df['Default_Payment'].astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to handle imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
#%%




#%% Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_resampled, y_resampled)
y_pred_lr = lr.predict(X_test_scaled)

print("Logistic Regression Report:\n", classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("AUC:", roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:,1]))

RocCurveDisplay.from_estimator(lr, X_test_scaled, y_test)
plt.title("ROC Curve - Logistic Regression")
plt.show()


print("Intercept:", lr.intercept_)
print("Coefficients:")
for feature, coef in zip(X.columns, lr.coef_[0]):
    print(f"{feature}: {coef:.4f}")


#%%
# Logistic Regression Interpretation
# Accuracy: 74% — overall good, but could be better.
# Precision (Class 1): 43% many false alarms predicting defaulters.

# Recall (Class 1): 57% it catches a bit over half of the actual defaulters.
# F1 Score (Class 1): 49% shows moderate performance in identifying defaults.
# AUC = 0.71 the model has fair ability to separate defaulters from non-defaulters.

# Summary: The model is better at predicting non-defaulters (Class 0). It needs improvement to better catch defaulters, possibly with a more powerful model.
#%%
 


#%%  Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_resampled, y_resampled)
y_pred_rf = rf.predict(X_test_scaled)

print(" Random Forest Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("AUC:", roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:,1]))

RocCurveDisplay.from_estimator(rf, X_test_scaled, y_test)
plt.title("ROC Curve - Random Forest")
plt.show()

#%%
# Random Forest Interpretation
# Accuracy: 79% — better overall than logistic regression.
# Precision (Class 1): 54% — fewer false positives than logistic regression.
# Recall (Class 1): 46% — slightly lower than logistic, misses more actual defaulters.
# F1 Score (Class 1): 50% — similar to logistic, still moderate.
# AUC = 0.75 — slightly better at distinguishing defaulters vs non-defaulters.

# Summary: Random Forest gives higher accuracy and does better at predicting non-defaulters. It's more balanced than logistic regression but still needs help detecting defaulters.





#%% XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=3)
xgb.fit(X_resampled, y_resampled)
y_pred_xgb = xgb.predict(X_test_scaled)

print(" XGBoost Report:\n", classification_report(y_test, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
print("AUC:", roc_auc_score(y_test, xgb.predict_proba(X_test_scaled)[:,1]))

RocCurveDisplay.from_estimator(xgb, X_test_scaled, y_test)
plt.title("ROC Curve - XGBoost")
plt.show()


#%%
# XGBoost Interpretation
# Accuracy: 71%, slightly lower than Random Forest.
# Precision (Class 1): 40%, many false positives, but acceptable.
# Recall (Class 1): 62%, best at catching defaulters among all models.
# F1 Score (Class 1): 48%, balanced result.
# AUC = 0.75, good separation ability between classes.

# Summary: XGBoost is best at finding defaulters (highest recall), though it trades off precision. Great choice if our priority is to catch more risky clients, even with a few false alarms.

#%%

#%%  SHAP Explanation (for XGBoost)
explainer = shap.Explainer(xgb)
shap_values = explainer(X)

shap.summary_plot(shap_values, X, plot_type='bar')
shap.summary_plot(shap_values, X)

#%%  Save models
joblib.dump(lr, "logistic_model.pkl")
joblib.dump(rf, "random_forest_model.pkl")
joblib.dump(xgb, "xgboost_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print(" Models and Scaler saved successfully.")
