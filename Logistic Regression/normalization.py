import numpy as np
class Normalizer:
    
    def __init__(self, mu=None, std=None):
        self.mu = mu
        self.std = std
    def normalize(self, X, mu=None, std=None):

        """
        Normalize data to z-scores using mean and standard deviation.
        
        Parameters:
        X (array-like): Input data to normalize.
        mu (float, optional): Mean to use. If None, computed from X.
        std (float, optional): Standard deviation to use (population std, ddof=0). If None, computed from X.
        
        Returns:
        tuple: (z-scores, mean of z-scores, variance of z-scores with ddof=0)
        
        Raises:
        ValueError: If std is 0 or input is invalid.
            """
        self.X = np.asarray(X)
        if self.X.size == 0:
            raise ValueError("Input array is empty")
        if np.isnan(self.X).any():
            raise ValueError("Input contains NaN values")
            
        if self.mu is None:
            self.mu = np.mean(X)
        if self.std is None:
            self.std = np.std(X, ddof=0)
        
        if self.std == 0:
            raise ValueError("Standard deviation is zero; cannot normalize")
            
        self.z_scores = (self.X - self.mu) / self.std
        return self.z_scores