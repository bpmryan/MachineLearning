import numpy as np 
from data import X, y, X_, y_
from bayes import NaiveBayes

model = NaiveBayes(X, y)
if __name__ == '__main__':
    # class, sex, age, fare
    print(model.predict(np.array([3, 1, 20, 2])))
    preds = [model.predict(x) for x in X_]
    accuracy = np.sum(preds == y_) / y_.shape[0]
    print(f'Accuracy: {accuracy}')