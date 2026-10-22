import pandas as pd

#LOAD DATASET
df = pd.read_csv("data/raw/PS_20174392719_1491204439457_log.csv")

#REMOVE ZERO AMOUNT
df = df[df["amount"] > 0]

#REMOVE DUPLICATES
df = df.drop_duplicates()

#RESET INDEX
df = df.reset_index(drop=True)

#SAVE CLEAN DATASET
df.to_csv("data/processed/paysim_clean.csv", index=False)

#CHECK RESULT
print("Clean Rows:", len(df))
print("Removed Rows:", 6362620 - len(df))
print("Missing Values:", df.isnull().sum().sum())
print("Duplicates:", df.duplicated().sum())
print("Clean Dataset Shape:", df.shape)