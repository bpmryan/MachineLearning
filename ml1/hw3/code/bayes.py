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
        posteriors = {c: self.pr_y_given_x(c, x) for c in self.classes}
        return max(posteriors, key=posteriors.get)
    
    def pr_y_given_x(self, y, x):
        prior = self.pr_y(y)
        likelihood = np.prod([self.pr_xi_given_y(xi, i, y) for i, xi in enumerate(x)])
        return prior * likelihood
    
    def pr_xi_given_y(self, xi, i, y):
        # likelihood: P(xi|y)
        counts_dict = self.feature_counts[y][i]
        count = counts_dict.get(xi, 0)
        total = sum(counts_dict.values())
        num_values = len(set(self.X[:, i]))
        return (count + 1) / (total + num_values)
        
    def pr_y(self, y):
        # prior probability P(y)
        return self.priors[y]

model = NaiveBayes(X, y)
if __name__ == '__main__':
    preds = [model.predict(x) for x in X_]
    accuracy = np.sum(preds == y_) / y_.shape[0]
    print(f'Accuracy: {accuracy}')
