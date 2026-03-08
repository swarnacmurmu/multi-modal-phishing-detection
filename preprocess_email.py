import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

data = pd.read_csv("data/raw/email_dataset.csv")

data['text_combined'] = data['text_combined'].str.lower()

data = data.dropna(subset=['text_combined'])

data.to_csv("data/processed/email_clean.csv", index=False)