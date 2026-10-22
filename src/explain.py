import pandas as pd

# Load Data
df = pd.read_csv("data/processed/gcn_xgboost_data.csv")

# Select Top 5 Suspicious Transactions
top_5 = df.sort_values(
    "gcn_risk_score",
    ascending=False
).head(5)

# Display Top 5
print("\nTOP 5 SUSPICIOUS TRANSACTIONS")


for i, (_, row) in enumerate(top_5.iterrows(), 1):

    print(f"\nTransaction {i}")

    print("Sender:", row["nameOrig"])
    print("Receiver:", row["nameDest"])
    print("Amount:", row["amount"])
    print("Risk Score:", round(row["gcn_risk_score"], 3))
    print("Outgoing Count:", row["outgoing_count"])
    print("Incoming Count:", row["incoming_count"])
    print("Account Activity:", row["account_activity"])
    print("Transaction Velocity:", round(row["txn_velocity"], 3))
    print("Actual Fraud Label:", row["isFraud"])