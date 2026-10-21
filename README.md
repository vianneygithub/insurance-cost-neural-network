# Neural Network for Tabular Regression

A PyTorch feed-forward neural network that predicts **medical insurance charges** from age, sex, BMI, number of children, smoker status and region. Built in a Jupyter notebook using the Kaggle [Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance).

## What it does

- Loads and explores the CSV
- Cleans and encodes the features, then splits into train, validation and test sets
- Trains a baseline linear regression and a neural network
- Reports RMSE, MAE and R² on the held-out test set
- Saves the trained model and preprocessing objects

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data

1. Create a Kaggle API token (Kaggle Settings, API, "Create New Token") and place `kaggle.json` in `~/.kaggle/` (on Windows: `C:\Users\<you>\.kaggle\`).
2. Download and unzip the data:

```bash
kaggle datasets download -d mirichoi0218/insurance --unzip -p data/
```

The notebook expects `data/insurance.csv`. To use another dataset, edit the config cell (`DATA_PATH`, `TARGET`, `ID_COLS`).

## Run

```bash
jupyter notebook neural_network_regression.ipynb
```

Run all cells from top to bottom.

## Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
Linear regression      RMSE:     7,793.21   MAE:     3,957.03   R2: 0.5995
Neural network         RMSE:     4,591.89   MAE:     2,336.37   R2: 0.8610
(np.float64(4591.893819739817), 2336.3700161283427, 0.8609587650695469)
## Project structure

```
neural_network_regression.ipynb   main notebook
requirements.txt
data/                             not committed, download from Kaggle
models/                           not committed, created by the notebook
```
