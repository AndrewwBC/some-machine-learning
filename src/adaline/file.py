import numpy as np

class Adaline:
    
    def __init__(self, eta=0.01, n_iter=25, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def net_input(self, x):
        return np.dot(x,self.w_) + self.b_   

    def activation(self, x):
        return x
    
    def predict(self, x):
        predict_list = []
        for feature in x:
            z = self.net_input(feature)
            predict_list.append(np.where(z >= 0.5, 1, 0))
            print(predict_list)
        return predict_list
    
    def fit(self, features, targets):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.1, scale=0.01, size=features.shape[1])
        self.b_ = np.float64(.0)
        n = len(features)
        
        for _ in range(self.n_iter):
            net_input = self.net_input(features)
            errors = targets - self.activation(net_input)
            self.w_ += self.eta * ( 2.0 * features.T.dot(errors) / n)
            self.b_ += self.eta * 2.0 * errors.mean()
            