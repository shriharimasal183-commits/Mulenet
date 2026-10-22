import pandas as pd

#LOAD DATASET
df = pd.read_csv("data/processed/paysim_clean.csv")

#CREATE TRANSACTION FEATURES
df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
df["is_cash_out"] = (df["type"] == "CASH_OUT").astype(int)

#SENDER FEATURES
sender = df.groupby("nameOrig").agg(
    outgoing_count=("nameOrig", "size"),
    outgoing_amount=("amount", "sum"),
    outgoing_avg=("amount", "mean"),
    first_step=("step", "min"),
    last_step=("step", "max")
).reset_index()

sender["active_steps"] = sender["last_step"] - sender["first_step"] + 1
sender["txn_velocity"] = sender["outgoing_count"] / sender["active_steps"]

#RECEIVER FEATURES
receiver = df.groupby("nameDest").agg(
    incoming_count=("nameDest", "size"),
    incoming_amount=("amount", "sum"),
    incoming_avg=("amount", "mean")
).reset_index()

#MERGE SENDER FEATURES
df = df.merge(
    sender,
    on="nameOrig",
    how="left"
)

#MERGE RECEIVER FEATURES
df = df.merge(
    receiver,
    on="nameDest",
    how="left"
)

#MONEY FLOW RATIO
df["money_flow_ratio"] = (
    df["outgoing_amount"] /
    (df["incoming_amount"] + 1)
)

#ACCOUNT ACTIVITY
df["account_activity"] = (
    df["outgoing_count"] +
    df["incoming_count"]
)

#SELECT FEATURES
features = df[
    [
        "step",
        "amount",
        "is_transfer",
        "is_cash_out",
        "outgoing_count",
        "incoming_count",
        "outgoing_amount",
        "incoming_amount",
        "outgoing_avg",
        "incoming_avg",
        "txn_velocity",
        "money_flow_ratio",
        "account_activity",
        "nameOrig",
        "nameDest",
        "isFraud"
    ]
]

#SAVE FEATURES
features.to_csv(
    "data/processed/paysim_features.csv",
    index=False
)

#CHECK RESULT
print("Feature Dataset Shape:", features.shape)

print("\nFeatures:")
print(features.columns.tolist())

print("\nFirst 5 Rows:")
print(features.head())

print("\nMissing Values:")
print(features.isnull().sum().sum())