import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from gnn_data import data


# CREATE TRAIN TEST MASK
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


# CREATE MODEL
class GAT(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = GATConv(
            5,
            32,
            heads=2,
            concat=True
        )

        self.conv2 = GATConv(
            64,
            16,
            heads=2,
            concat=True
        )

        self.conv3 = GATConv(
            32,
            2,
            heads=1,
            concat=False
        )

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.elu(x)

        x = self.conv2(x, edge_index)
        x = F.elu(x)

        x = self.conv3(x, edge_index)

        return x


# CREATE MODEL
model = GAT()


# CLASS WEIGHTS
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


# OPTIMIZER
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


# TRAIN MODEL
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
            "Epoch:",
            epoch,
            "Loss:",
            loss.item()
        )


# PREDICT
model.eval()

with torch.no_grad():

    output = model(
        data.x,
        data.edge_index
    )

    prediction = output.argmax(dim=1)


# TEST RESULTS
y_test = data.y[test_mask]
y_pred = prediction[test_mask]


# EVALUATE
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


print("\nGAT Test Results")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print("\nTraining Nodes:", train_mask.sum().item())
print("Testing Nodes:", test_mask.sum().item())