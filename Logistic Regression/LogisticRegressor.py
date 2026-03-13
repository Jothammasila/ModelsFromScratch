import numpy as np
from normalization import Normalizer
import matplotlib.pyplot as plt

class LogisticRegressor:

    def __init__(self, learning_rate=0.001, num_iterations=1000):
        self.learning_rate = learning_rate
        self.normalizer = Normalizer()
        self.num_iterations = num_iterations

# Activation function
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

# Binary Cross-Entropy Loss function implemenation
    def loss(self, y_true, y_pred):
        n = len(y_true)
        y_pred = np.clip(y_pred,1e-9,1-1e-9)
        return -(1/n)*np.sum(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))

    def train(self, X, y):

        self.X = X
        self.y_true = y

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        self.loss_history = []

        for i in range(self.num_iterations):

            linear_model = np.dot(X,self.weights)+self.bias
            y_pred = self.sigmoid(linear_model)

            self.L = self.loss(self.y_true, y_pred)
            self.loss_history.append(self.L)

            dw = (1/n_samples)*np.dot(X.T,(y_pred-self.y_true))
            db = (1/n_samples)*np.sum(y_pred-self.y_true)

            self.weights -= self.learning_rate*dw
            self.bias -= self.learning_rate*db

            if (i+1)%1000==0:
                print(f"Iter {i+1} Loss {self.L:.4f}")

    def predict(self,X):
        linear_model = np.dot(X,self.weights)+self.bias
        y_pred = self.sigmoid(linear_model)
        return (y_pred>0.5).astype(int)

    def parameter_summary(self, show_loss=False):
        print(f"Weights: {self.weights}")
        print(f"Bias: {self.bias}")
        
        if show_loss:
            if hasattr(self, "loss_history") and len(self.loss_history) > 0:
                plt.figure(figsize=(8,5))
                plt.plot(self.loss_history)
                plt.xlabel("Iteration")
                plt.ylabel("Loss")
                plt.title("Training Loss Curve")
                plt.grid(True)
                plt.show()
        else:
            print("Loss history is empty. Train the model first.")