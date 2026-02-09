import numpy as np
from data import X, y, X_, y_

class NaiveBayes:
    def __init__(self, X, y):
        self.X = X # X is a matrix   
        self.y = y # y is a vector
        self.classes = np.unique(y)
        # store data grouped in class to make probability calculations
        self.priors = {c: np.mean(y == c) for c in self.classes}

        # store the counts of each class
        self.feature_counts = {}
        for c in self.classes:
            X_c = X[y == c]
            self.feature_counts[c] = []
            for i in range(X.shape[1]):
                values, counts = np.unique(X_c[:, i], return_counts=True)
                self.feature_counts[c].append(dict(zip(values, counts)))
        pass
        
    def predict(self, x):
        # x is another vector
        loggies = np.zeros(x.shape)
        for i, xi in enumerate(x):
            loggies[i] = np.log(self.pr_xi_given_y(xi, i,  y))
        return np.sum(loggies) + np.log(self.pr_y(y))    
    
    def pr_y_given_x(self, y, x):
        prior = self.pr_y(y)
        likelihood = np.prod([self.pr_xi_given_y(xi, i, y) for i, xi in enumerate(x)])
        return prior * likelihood
    
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
    preds = [model.predict(x) for x in X_]
    accuracy = np.sum(preds == y_) / y_.shape[0]
    print(f'Accuracy: {accuracy}')
