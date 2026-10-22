import pandas as pd
import torch
from torch_geometric.data import Data

#LOAD DATASET
df = pd.read_csv("data/processed/graph_data.csv")

#CREATE NODE LIST
nodes = pd.unique(
    pd.concat([df["nameOrig"], df["nameDest"]])
)

#CREATE NODE ID
node_id = {
    node: i
    for i, node in enumerate(nodes)
}

#CREATE ACCOUNT FEATURES
outgoing_count = df.groupby("nameOrig").size()
incoming_count = df.groupby("nameDest").size()

outgoing_amount = df.groupby("nameOrig")["amount"].sum()
incoming_amount = df.groupby("nameDest")["amount"].sum()

#CREATE NODE FEATURES
node_features = []

for node in nodes:

    out_count = outgoing_count.get(node, 0)
    in_count = incoming_count.get(node, 0)

    out_amount = outgoing_amount.get(node, 0)
    in_amount = incoming_amount.get(node, 0)

    activity = out_count + in_count

    node_features.append([
        out_count,
        in_count,
        out_amount,
        in_amount,
        activity
    ])

node_features = torch.tensor(
    node_features,
    dtype=torch.float
)

#NORMALIZE FEATURES
node_features = (
    node_features - node_features.mean(dim=0)
) / (
    node_features.std(dim=0) + 1e-8
)

#CREATE EDGES
source = df["nameOrig"].map(node_id).values
target = df["nameDest"].map(node_id).values

edge_index = torch.tensor(
    [source, target],
    dtype=torch.long
)

#CREATE NODE LABELS
labels = torch.zeros(
    len(nodes),
    dtype=torch.long
)

#LABEL FRAUD NODES
for _, row in df[df["isFraud"] == 1].iterrows():

    sender = node_id[row["nameOrig"]]

    labels[sender] = 1

#CREATE GRAPH
data = Data(
    x=node_features,
    edge_index=edge_index,
    y=labels
)

#CHECK GRAPH
print("Nodes:", data.num_nodes)
print("Edges:", data.num_edges)
print("Node Features:", data.x.shape)
print("Labels:", data.y.shape)