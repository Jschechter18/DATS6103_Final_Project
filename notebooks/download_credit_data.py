#%%
import pandas as pd

df = pd.read_csv("credit_card_clients_full.csv")

# Display the number of rows and the first few entries
print(f"Total rows: {len(df)}")
print(df.head())

# %%

# df.info()
print(df.isnull().sum())
# %%


