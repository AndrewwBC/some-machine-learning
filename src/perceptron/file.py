import numpy as np

class Perceptron:
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def network_input(self, feature):
        return np.dot(feature, self.weights) + self.bias
    
    def fit(self, features, targets):
        rgen = np.random.RandomState(self.random_state)
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=features.shape[1])
        self.bias = np.float64(0.)
        
        for _ in range(self.n_iter):
            self.predictions = []
            
            for xi, target in zip(features, targets):
                predicted = self.predicted(xi)
                update = self.eta * (target - predicted)
                
                self.predictions.append(predicted)
                self.weights += update *  xi
                self.bias += update
       
        return self
    
    def predicted(self, x):
        z = self.network_input(x)
        print(z)
        return np.where(z >= 0.0, 1, 0)
    
    def predict(self, x):
        predict_list = []
        for row in x:
            z = self.network_input(row)
            predict_list.append(np.where(z >= 0.0, 1, 0))
        return predict_list
        