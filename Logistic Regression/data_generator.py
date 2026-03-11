import numpy as np
import pandas as pd

"""
Data generator for creating a binary classification dataset. 
This file can be run independently to generate a CSV file with two columns: 'value' and 'label'. 
The 'value' column contains random integers between 80 and 125, while the 'label' column contains binary labels indicating whether the corresponding value is greater than or equal to 100 (1) or less than 100 (0). 
The generated dataset can be used for training and testing the Logistic Regressor model implemented in LogisticRegressor.py.
"""

np.random.seed(42)

values = np.random.randint(80, 125, 1000000)
labels = (values >= 100).astype(int)

df = pd.DataFrame({
    "value": values,
    "label": labels
})

df.to_csv("./data.csv", index=False)