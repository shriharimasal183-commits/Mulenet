
# Mule Account Network Detection in UPI Payments Using Graph Neural Networks

## About
This project analyzes suspicious UPI transaction patterns using traditional Machine Learning and Graph Neural Networks (GNNs).

Accounts are represented as nodes, and transactions are represented as edges. Graph-based analysis helps investigate suspicious transaction relationships.

## Technologies
- Python
- Pandas and NumPy
- Scikit-learn
- PyTorch and PyTorch Geometric
- NetworkX
- Streamlit

## Models
### Traditional Machine Learning
- Logistic Regression
- Random Forest
- XGBoost

### Graph Neural Networks
- GCN
- GraphSAGE
- GAT

## Project Structure
- `src/`: Data processing, feature engineering, and model scripts
- `dashboard/`: Streamlit dashboard
- `outputs/`: Network visualization output

## Installation

Clone the repository:

```bash
git clone https://github.com/shriharimasal183-commits/Mulenet.git
cd Mulenet
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Note: Create `requirements.txt` before running the installation command.

## Dataset
The dataset is not included in this repository because of its large size.

Place the required dataset in `data/raw/` and follow the preprocessing steps in the source scripts.

## Run the Dashboard

```bash
streamlit run dashboard/app.py
```

## Evaluation
Model evaluation scripts are included in `src/`. Performance results will be documented after verifying the current model outputs.

## Future Improvements
- Improve suspicious transaction explanations
- Evaluate models consistently
- Explore real-time transaction monitoring

## Disclaimer
This is an academic research project. Its outputs are experimental and do not prove that a transaction or account is fraudulent.
