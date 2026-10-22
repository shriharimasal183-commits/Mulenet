# MODEL COMPARISON

results = {
    "Random Forest": {
        "Accuracy": 0.9465767635,
        "Precision": 0.8952916374,
        "Recall": 0.7768292683,
        "F1 Score": 0.8318641854
    },

    "XGBoost": {
        "Accuracy": 0.9548755187,
        "Precision": 0.9163787146,
        "Recall": 0.8085365854,
        "F1 Score": 0.8590864917
    },

    "GraphSAGE": {
        "Accuracy": 0.7757674860,
        "Precision": 0.2577377603,
        "Recall": 0.8237410072,
        "F1 Score": 0.3926275182
    },

    "GCN": {
        "Accuracy": 0.7873193375,
        "Precision": 0.2651033386,
        "Recall": 0.7997601918,
        "F1 Score": 0.3982089552
    },

    "GAT": {
        "Accuracy": 0.7820972676,
        "Precision": 0.2578171091,
        "Recall": 0.7859712230,
        "F1 Score": 0.3882718792
    }
}


# PRINT COMPARISON

print("\nMODEL COMPARISON")
print("-" * 75)

print(
    f"{'Model':<15}"
    f"{'Accuracy':<15}"
    f"{'Precision':<15}"
    f"{'Recall':<15}"
    f"{'F1 Score':<15}"
)

print("-" * 75)

for model, metrics in results.items():

    print(
        f"{model:<15}"
        f"{metrics['Accuracy']*100:<15.2f}"
        f"{metrics['Precision']*100:<15.2f}"
        f"{metrics['Recall']*100:<15.2f}"
        f"{metrics['F1 Score']*100:<15.2f}"
    )