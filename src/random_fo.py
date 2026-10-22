import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#LOAD DATASET
df = pd.read_csv("data/processed/graph_data.csv")

#SELECT FEATURES
X = df[
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
        "account_activity"
    ]
]

#SELECT TARGET
y = df["isFraud"]

#SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#CREATE MODEL
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

#TRAIN MODEL
model.fit(X_train, y_train)

#PREDICT
y_pred = model.predict(X_test)

#EVALUATE
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))