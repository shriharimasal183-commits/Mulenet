import pandas as pd

#LOAD DATASET
df = pd.read_csv("data/processed/paysim_features.csv")

#SEPARATE FRAUD
fraud = df[df["isFraud"] == 1]

#SEPARATE NORMAL
normal = df[df["isFraud"] == 0]

#SAMPLE NORMAL
normal_sample = normal.sample(
    n=40000,
    random_state=42
)

#COMBINE DATA
graph_df = pd.concat(
    [fraud, normal_sample]
).sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

#SAVE GRAPH DATA
graph_df.to_csv(
    "data/processed/graph_data.csv",
    index=False
)

#CHECK RESULT
print("Fraud Transactions:", len(fraud))
print("Normal Transactions Used:", len(normal_sample))
print("Graph Transactions:", len(graph_df))

print("\nFraud Distribution:")
print(graph_df["isFraud"].value_counts())

print("\nUnique Senders:", graph_df["nameOrig"].nunique())
print("Unique Receivers:", graph_df["nameDest"].nunique())

nodes = set(graph_df["nameOrig"]) | set(graph_df["nameDest"])

print("Total Graph Nodes:", len(nodes))