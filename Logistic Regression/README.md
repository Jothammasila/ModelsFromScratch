# Logistic Regression from Scratch

This folder contains a Python implementation of Logistic Regression without using scikit-learn.

## Files
- `data_generator.py` – Generates random binary classification dataset (the data is saved in `data.csv` file)
- `train.py` – Training script
- `LogisticRegressor.py` – Implementation of Logistic Regression class
- `normalization.py` - Implementation of the data normalization function.
- `model.py` - Model training script (the model is saved in a `logistic_regressor_model.pkl`)
- `test_model.py` - Loading and testing the trained model

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Mathematical Derivation
The gradient derivation for logistic regression used in this project
can be found in:

Loss_function_and_Weights_Update_theory.pdf