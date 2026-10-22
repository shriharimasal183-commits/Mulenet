import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from xgboost import XGBClassifier


#LOAD GRAPH
from gnn_data import data

print("Graph Loaded")
print("Nodes:", data.num_nodes)
print("Edges:", data.num_edges)



#CREATE SAME TRAIN/TEST NODE SPLIT
torch.manual_seed(42)

num_nodes = data.num_nodes

indices = torch.randperm(num_nodes)

train_size = int(0.8 * num_nodes)

train_mask = torch.zeros(
    num_nodes,
    dtype=torch.bool
)

test_mask = torch.zeros(
    num_nodes,
    dtype=torch.bool
)

train_mask[indices[:train_size]] = True
test_mask[indices[train_size:]] = True


# CREATE GCN
class GCN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(5, 32)
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, 2)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)

        return x



#TRAIN GCN
model = GCN()

train_labels = data.y[train_mask]

normal_count = (train_labels == 0).sum().item()
fraud_count = (train_labels == 1).sum().item()

class_weights = torch.tensor(
    [
        1.0,
        normal_count / fraud_count
    ],
    dtype=torch.float
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


for epoch in range(1, 51):

    model.train()

    optimizer.zero_grad()

    output = model(
        data.x,
        data.edge_index
    )

    loss = F.cross_entropy(
        output[train_mask],
        data.y[train_mask],
        weight=class_weights
    )

    loss.backward()

    optimizer.step()

    if epoch % 10 == 0:

        print(
            "GCN Epoch:",
            epoch,
            "Loss:",
            loss.item()
        )



# GET GCN RISK SCORE


model.eval()

with torch.no_grad():

    gcn_output = model(
        data.x,
        data.edge_index
    )

    # Probability of fraud
    gcn_probability = torch.softmax(
        gcn_output,
        dim=1
    )[:, 1]


gcn_risk_score = gcn_probability.numpy()


print("\nGCN Risk Scores Created")

print(
    "Minimum:",
    gcn_risk_score.min()
)

print(
    "Maximum:",
    gcn_risk_score.max()
)



#LOAD TRANSACTION DATA

df = pd.read_csv(
    "data/processed/graph_data.csv"
)



# MAP GCN RISK SCORE TO TRANSACTIONS
nodes = pd.unique(
    pd.concat(
        [
            df["nameOrig"],
            df["nameDest"]
        ]
    )
)

node_id = {
    node: i
    for i, node in enumerate(nodes)
}


df["sender_gcn_risk"] = (
    df["nameOrig"]
    .map(node_id)
    .map(
        lambda x: gcn_risk_score[x]
    )
)


df["receiver_gcn_risk"] = (
    df["nameDest"]
    .map(node_id)
    .map(
        lambda x: gcn_risk_score[x]
    )
)


# Combined network risk
df["gcn_risk_score"] = (
    df["sender_gcn_risk"] +
    df["receiver_gcn_risk"]
) / 2



# CREATE XGBOOST FEATURES
features = [
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
    "gcn_risk_score"
]


X = df[features]

y = df["isFraud"]

#TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#TRAIN XGBOOST
model_xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)


model_xgb.fit(
    X_train,
    y_train
)


# PREDICTION

y_pred = model_xgb.predict(
    X_test
)

# EVALUATION
accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print(
    "Accuracy:",
    accuracy
)

print(
    "Precision:",
    precision
)

print(
    "Recall:",
    recall
)

print(
    "F1 Score:",
    f1
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

df.to_csv(
    "data/processed/gcn_xgboost_data.csv",
    index=False
)