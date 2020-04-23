import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt 

# Load the data
data = pd.read_csv('data\\ex2data1.txt', header = None)
X = data.iloc[:,:-1]
y = data.iloc[:,2]
print(data.head())

mask = y == 1 # If the element in y == 1 then set the corresponding element in mask to True (or 1)


print(mask)
adm = plt.scatter(X[mask][0].values, X[mask][1].values)
not_adm = plt.scatter(X[~mask][0].values, X[~mask][1].values)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend((adm, not_adm), ('Admitted', 'Not admitted'))
plt.show()