import numpy as np
from sklearn import datasets

class Perceptron:
    
    def __init__(self, data, n_iter, target): 
        rgen = np.random.RandomState(42)
        self.n_iter = n_iter
        self.target = target
        self.data = data
        self.features = self.data.drop(columns = self.target).values
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=data.drop(columns = target).shape[1])
        self.bias = rgen.random()
        
        
    def network_input(self, xi):
        return np.dot(xi, self.weights) + self.bias
    
    def fit(self):
        eta = 0.1
        target_values = self.data[self.target].values
        
        for _ in range(self.n_iter):
            
            for xi, target in zip(self.features, target_values):
                predicted = self.predict(xi)
                update = eta * (target - predicted)
                print(f"Previu:{predicted} - Alvo: {target}")
                self.weights += update *  xi
                self.bias += update
    
    def predict(self, x):
        z = self.network_input(x)
        return np.where(z >= 0.0, 1, 0)
  


iris = datasets.load_iris(as_frame=True)
data = iris.frame[iris.feature_names[:2] + ['target']]
data = data[data['target'] != 2]  # binário 0/1

p = Perceptron(data, n_iter=10, target='target')
p.fit()

# Predição
print(p.predict(data.drop(columns='target').values[3]))
