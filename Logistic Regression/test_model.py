import pickle
from data_preprocessing import DataPreprocessor


# Load the trained model from the pickle file
with open("./logistic_regressor_model.pkl", "rb") as f:
    model = pickle.load(f)

# Print the model parameters
model.parameter_summary()

# Load the testing data using the DataPreprocessor
preprocessor = DataPreprocessor()
_, _, X_test, y_test = preprocessor.preprocess()

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = (y_pred == y_test).mean()
print(f"Test Accuracy: {accuracy:.4f}")