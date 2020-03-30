import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

#load the data
X = []
Y = []
for line in open('data_2d.csv'):
    x1,x2,y=line.split(',')
    X.append([1,float(x1),float(x2)])
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

X_transpose = X.transpose()
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
y_pred = np.dot(X, w)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
plt.show()