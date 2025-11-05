import numpy as np

class Perceptron:
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def network_input(self, feature):
        return np.dot(self.weights, feature) + self.bias
    
    def fit(self, features, targets):
        rgen = np.random.RandomState(self.random_state)
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=features.shape[1])
        self.bias = rgen.random()
        
        for _ in range(self.n_iter):
            self.predictions = []
            for xi, target in zip(features, targets):
                predicted = self.predict(xi)
                update = self.eta * (target - predicted)
                self.predictions.append(predicted)
                self.weights += update *  xi
                self.bias += update
       
        return self
    
    def predict(self, x):
        z = self.network_input(x)
        return np.where(z >= 0.0, 1, 0)