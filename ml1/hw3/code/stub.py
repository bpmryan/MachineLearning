import numpy as np
from data import X, y, X_, y_

class NaiveBayes:
    def __init__(self, X, y):
        pass
    def predict(self, x):
        pass
    def pr_y_given_x(self, y, x):
        pass
    def pr_xi_given_y(self, xi, i, y):
        pass
    def pr_y(self, y):
        pass

model = NaiveBayes(X, y)
if __name__ == '__main__':
    preds = [model.predict(x) for x in X_]
    accuracy = np.sum(preds == y_) / y_.shape[0]
    print(f'Accuracy: {accuracy}')
