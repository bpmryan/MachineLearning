import numpy as np 
from data import X, y 
from bayes import NaiveBayes

model = NaiveBayes(X, y)

# np.array([class, sex, age, fare (in lbs)])
testSubject = np.array([3, 1, 20, 20])

prediction = model.predict(testSubject)

if prediction == 1:
    print("You survived")
else:
    print("You've somehow went further than 6ft under. Whomp whomp o7")
    print("Better luck next time")