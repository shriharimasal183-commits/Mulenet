import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv("data/processed/gcn_xgboost_data.csv")

# Select High Risk Transactions
df = df.sort_values(
    "gcn_risk_score",
    ascending=False
).head(80)

# Create Graph
G = nx.DiGraph()

for _, row in df.iterrows():

    sender = row["nameOrig"]
    receiver = row["nameDest"]

    G.add_edge(
        sender,
        receiver,
        amount=row["amount"],
        risk=row["gcn_risk_score"]
    )

# Remove Isolated Components
components = sorted(
    nx.weakly_connected_components(G),
    key=len,
    reverse=True
)

# Keep Main Connected Components
selected_nodes = set()

for component in components[:8]:
    selected_nodes.update(component)

G = G.subgraph(selected_nodes).copy()

# Calculate Node Risk
for node in G.nodes():

    risks = []

    for _, _, data in G.in_edges(node, data=True):
        risks.append(data["risk"])

    for _, _, data in G.out_edges(node, data=True):
        risks.append(data["risk"])

    G.nodes[node]["risk"] = max(risks) if risks else 0

# Create Positions
pos = nx.spring_layout(
    G,
    seed=42,
    k=1.5,
    iterations=150
)

# Create Figure
fig, ax = plt.subplots(figsize=(16, 10))

# Draw Edges
nx.draw_networkx_edges(
    G,
    pos,
    ax=ax,
    edge_color="gray",
    width=1.2,
    alpha=0.7,
    arrows=True,
    arrowsize=18,
    arrowstyle="-|>",
    min_source_margin=10,
    min_target_margin=12
)

# Normal Nodes
normal_nodes = [
    node for node in G.nodes()
    if G.nodes[node]["risk"] < 0.7
]

nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=normal_nodes,
    node_color="blue",
    node_size=180,
    alpha=0.8,
    ax=ax
)

# High Risk Nodes
red_nodes = [
    node for node in G.nodes()
    if G.nodes[node]["risk"] >= 0.7
]

nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=red_nodes,
    node_color="red",
    node_size=350,
    alpha=0.95,
    ax=ax
)

# Node Labels
nx.draw_networkx_labels(
    G,
    pos,
    labels={
        node: node
        for node in G.nodes()
    },
    font_size=6,
    font_weight="bold",
    ax=ax
)

# Small Red Account Panel
top_red_nodes = sorted(
    red_nodes,
    key=lambda n: G.nodes[n]["risk"],
    reverse=True
)[:5]

panel_text = "HIGH RISK ACCOUNTS\n"

for i, node in enumerate(top_red_nodes, 1):

    risk = G.nodes[node]["risk"]
    connections = G.degree(node)

    panel_text += (
        f"{i}. {node} | "
        f"Risk: {risk:.2f} | "
        f"Links: {connections}\n"
    )

ax.text(
    0.02,
    0.03,
    panel_text,
    transform=ax.transAxes,
    fontsize=8,
    verticalalignment="bottom",
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="white",
        edgecolor="gray",
        alpha=0.9
    )
)

# Hover Annotation
annotation = ax.annotate(
    "",
    xy=(0, 0),
    xytext=(15, 15),
    textcoords="offset points",
    fontsize=9,
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="white",
        edgecolor="gray",
        alpha=0.95
    )
)

annotation.set_visible(False)

# Hover Function
def hover(event):

    if event.inaxes != ax:
        return

    for node in G.nodes():

        x, y = pos[node]

        screen_x, screen_y = ax.transData.transform(
            (x, y)
        )

        distance = (
            (event.x - screen_x) ** 2 +
            (event.y - screen_y) ** 2
        ) ** 0.5

        if distance < 15:

            risk = G.nodes[node]["risk"]
            connections = G.degree(node)

            if risk >= 0.7:
                status = "HIGH RISK"
            else:
                status = "NORMAL"

            annotation.xy = (x, y)

            annotation.set_text(
                f"Account: {node}\n"
                f"Status: {status}\n"
                f"Risk Score: {risk:.3f}\n"
                f"Connections: {connections}"
            )

            annotation.set_visible(True)

            fig.canvas.draw_idle()

            return

    annotation.set_visible(False)

    fig.canvas.draw_idle()


# Connect Hover
fig.canvas.mpl_connect(
    "motion_notify_event",
    hover
)

# Title
plt.title(
    "UPI Transaction Network - GCN Risk Visualization",
    fontsize=16
)

# Legend
ax.scatter(
    [],
    [],
    s=100,
    color="blue",
    label="Normal Account"
)

ax.scatter(
    [],
    [],
    s=180,
    color="red",
    label="High Risk Account"
)

ax.legend(
    loc="upper right"
)

# Remove Axis
plt.axis("off")

plt.tight_layout()

plt.show()