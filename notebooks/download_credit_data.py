from datasets import load_dataset
import pandas as pd

# Load the dataset
dataset = load_dataset("scikit-learn/credit-card-clients", split='train')

# Convert to a pandas DataFrame
df = dataset.to_pandas()

# Display the number of rows and the first few entries
print(f"Total rows: {len(df)}")
print(df.head())

# Save the DataFrame to a CSV file
df.to_csv("../data/credit_card_clients_full.csv")