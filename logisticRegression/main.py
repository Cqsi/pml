import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt 

from costFunction import costFunction
from gradient import gradient

# Load the data
data = pd.read_csv('data\\ex2data1.txt', header = None)
X = data.iloc[:,:-1]
y = data.iloc[:,2]
#print(data.head())

mask = y == 1 # If the element in y == 1 then set the corresponding element in mask to True (or 1)

#adm = plt.scatter(X[mask][0].values, X[mask][1].values)
#not_adm = plt.scatter(X[~mask][0].values, X3[~mask][1].values)
#plt.xlabel('Exam 1 score')
#plt.ylabel('Exam 2 score')
#plt.legend((adm, not_adm), ('Admitted', 'Not admitted'))
#plt.show()

(m, n) = X.shape
X = np.hstack((np.ones((m,1)), X))
y = y[:, np.newaxis]
theta = np.zeros((n+1,1)) # intializing theta with all zeros
J = costFunction(theta, X, y)
print(J)

temp = opt.fmin_tnc(func = costFunction, x0 = theta.flatten(), fprime = gradient, args = (X, y.flatten()))
#the output of above function is a tuple whose first element #contains the optimized values of theta

theta_optimized = temp[0]
print(theta_optimized)