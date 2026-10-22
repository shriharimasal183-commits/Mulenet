import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

# Page Setup
st.set_page_config(
    page_title="UPI Fraud Network Detection",
    layout="wide"
)

# Load Data
df = pd.read_csv(
    "data/processed/gcn_xgboost_data.csv"
)

# Title
st.title("UPI Fraud Network Detection")


# Metrics
total_transactions = len(df)
suspicious_transactions = len(
    df[df["gcn_risk_score"] >= 0.5]
)
average_risk = df["gcn_risk_score"].mean()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Transactions",
    f"{total_transactions:,}"
)

col2.metric(
    "Suspicious Transactions",
    f"{suspicious_transactions:,}"
)

col3.metric(
    "Average Risk Score",
    f"{average_risk:.3f}"
)

st.divider()

# Top 5 Suspicious Transactions
st.subheader("Top 5 Suspicious Transactions")

top_5 = df.sort_values(
    "gcn_risk_score",
    ascending=False
).head(5)

top_5_display = top_5[
    [
        "nameOrig",
        "nameDest",
        "amount",
        "gcn_risk_score",
        "isFraud"
    ]
].copy()

top_5_display.columns = [
    "Sender",
    "Receiver",
    "Amount",
    "Risk Score",
    "Fraud Label"
]

top_5_display["Risk Score"] = (
    top_5_display["Risk Score"].round(3)
)

st.dataframe(
    top_5_display,
    use_container_width=True,
    hide_index=True
)

st.divider()

# Create Graph
graph_df = df.sort_values(
    "gcn_risk_score",
    ascending=False
).head(80)

G = nx.DiGraph()

for _, row in graph_df.iterrows():

    sender = row["nameOrig"]
    receiver = row["nameDest"]

    G.add_edge(
        sender,
        receiver,
        amount=row["amount"],
        risk=row["gcn_risk_score"]
    )

# Keep Main Components
components = sorted(
    nx.weakly_connected_components(G),
    key=len,
    reverse=True
)

selected_nodes = set()

for component in components[:8]:
    selected_nodes.update(component)

G = G.subgraph(
    selected_nodes
).copy()

# Calculate Node Risk
for node in G.nodes():

    risks = []

    for _, _, data in G.in_edges(node, data=True):
        risks.append(data["risk"])

    for _, _, data in G.out_edges(node, data=True):
        risks.append(data["risk"])

    G.nodes[node]["risk"] = (
        max(risks) if risks else 0
    )

# Create Positions
pos = nx.spring_layout(
    G,
    seed=42,
    k=1.5,
    iterations=120
)

# Graph
edge_x = []
edge_y = []
annotations = []

for source, target, data in G.edges(data=True):

    x1, y1 = pos[source]
    x2, y2 = pos[target]

    edge_x.extend([x1, x2, None])
    edge_y.extend([y1, y2, None])

    annotations.append(
        dict(
            x=x2,
            y=y2,
            ax=x1,
            ay=y1,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="gray"
        )
    )

edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    mode="lines",
    line=dict(
        width=1
    ),
    hoverinfo="none"
)

node_x = []
node_y = []
node_text = []
node_color = []
node_size = []

for node in G.nodes():

    x, y = pos[node]

    risk = G.nodes[node]["risk"]
    connections = G.degree(node)

    node_x.append(x)
    node_y.append(y)

    node_text.append(
        f"Account: {node}<br>"
        f"Risk Score: {risk:.3f}<br>"
        f"Connections: {connections}"
    )

    if risk >= 0.7:
        node_color.append("red")
        node_size.append(18)
    else:
        node_color.append("blue")
        node_size.append(10)

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers",
    text=node_text,
    hoverinfo="text",
    marker=dict(
        size=node_size,
        color=node_color,
        line=dict(
            width=1
        )
    )
)

# Display Graph
st.subheader("Suspicious Transaction Network")

fig = go.Figure(
    data=[
        edge_trace,
        node_trace
    ]
)

fig.update_layout(
    height=650,
    showlegend=False,
    hovermode="closest",
    margin=dict(
        l=10,
        r=10,
        t=10,
        b=10
    ),
    xaxis=dict(
        visible=False
    ),
    yaxis=dict(
        visible=False
    ),
    annotations=annotations
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# High Risk Accounts
st.subheader("High Risk Accounts")

red_accounts = []

for node in G.nodes():

    risk = G.nodes[node]["risk"]

    if risk >= 0.7:

        red_accounts.append(
            {
                "Account": node,
                "Risk Score": round(risk, 3),
                "Connections": G.degree(node)
            }
        )

red_df = pd.DataFrame(red_accounts)

red_df = red_df.sort_values(
    "Risk Score",
    ascending=False
).head(10)

if len(red_df) > 0:

    st.dataframe(
        red_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info("No high-risk accounts found.")

st.divider()

# Selected Transaction
st.subheader("Transaction Explanation")

selected_index = st.selectbox(
    "Select Transaction",
    range(len(top_5)),
    format_func=lambda x: f"Transaction {x + 1}"
)

row = top_5.iloc[selected_index]

st.write(
    f"**Sender:** {row['nameOrig']}"
)

st.write(
    f"**Receiver:** {row['nameDest']}"
)

st.write(
    f"**Amount:** {row['amount']:,.2f}"
)

st.write(
    f"**GCN Risk Score:** {row['gcn_risk_score']:.3f}"
)

# Risk Status
if row["gcn_risk_score"] >= 0.7:

    st.error("High Risk Transaction")

elif row["gcn_risk_score"] >= 0.5:

    st.warning("Suspicious Transaction")

else:

    st.success("Low Risk Transaction")