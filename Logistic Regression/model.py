from LogisticRegressor import LogisticRegressor
from data_preprocessing import DataPreprocessor
import pickle

# Initialize the data preprocessor
preprocessor = DataPreprocessor()
X_train, y_train, _, _ = preprocessor.preprocess()

# Initialize the logistic regression model
model = LogisticRegressor(learning_rate=0.01, num_iterations=10000)

# Train the model
model.train(X_train, y_train)

# Model parameter summary
model.parameter_summary()

# Save the trained model to a file using pickle
with open("./logistic_regressor_model.pkl", "wb") as f:
    pickle.dump(model, f)