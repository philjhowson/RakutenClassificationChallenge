import pandas as pd

df = pd.read_csv('data/processed/translated_text.csv')

df.to_parquet('translated_text.parquet')
