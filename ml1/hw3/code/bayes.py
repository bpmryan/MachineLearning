import numpy as np
from data import X, y, X_, y_

class NaiveBayes:
    def __init__(self, X, y):
        self.X = X # X is a matrix   
        self.y = y # y is a vector
        
    def predict(self, x):
        labels = np.unique(self.y)
        prob = [self.pr_y_given_x(y, x) for y in labels]
        return np.argmax(prob)
    
    def pr_y_given_x(self, y, x):
        # x is another vector
        loggies = np.zeros(x.shape)
        for i, xi in enumerate(x):
            loggies[i] = np.log(self.pr_xi_given_y(xi, i,  y))
        return np.sum(loggies) + np.log(self.pr_y(y)) 
    
    def pr_xi_given_y(self, xi, i, y):
        # likelihood: P(xi|y)
        filtered = self.X[self.y == y]
        count = np.sum(filtered[:, i] == xi) 
        probability = count / filtered.shape[0]
        return probability if probability > 0 else 2**(-32)
            
    def pr_y(self, y):
        # prior probability P(y)
        count = np.sum(self.y == y)
        total = self.y.shape[0]
        return count / total

model = NaiveBayes(X, y)
if __name__ == '__main__':
    print(model.predict(np.array([1, 1, 25, 2])))
    preds = [model.predict(x) for x in X_]
    accuracy = np.sum(preds == y_) / y_.shape[0]
    print(f'Accuracy: {accuracy}')
