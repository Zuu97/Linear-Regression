import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

#load the data
X = []
Y = []
for line in open('data_poly.csv'):
    x,y=line.split(',')
    X.append([1,float(x),float(x)**2])
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

X_transpose = X.transpose()
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
y_pred = np.dot(X, w)

plt.scatter(X[:,1],Y)
plt.plot(sorted(X[:,1]),sorted(y_pred))
plt.show()

d1 = Y - y_pred
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("the r-squared is:", r2)