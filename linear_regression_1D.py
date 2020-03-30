import numpy as np 
import matplotlib.pyplot as plt 

#load the data
X = []
Y = []
for line in open('data_1d.csv'):
    x,y=line.split(',')
    X.append(float(x))
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

data_size = len(X)

denominator = (data_size * np.dot(X,X)) - (np.sum(X) ** 2)
numerator_a = (data_size * np.dot(Y,X)) - (np.sum(X) * np.sum(Y))
numerator_b = (np.sum(X) * np.dot(Y,X)) - (np.sum(Y) * np.dot(X,X))  

#predict the model
a , b = numerator_a / denominator , numerator_b / (-1 * denominator)
y_pred = a * X + b

#Evaluate the model
SS_res = np.dot((Y - y_pred), (Y - y_pred))
SS_tot = np.dot((Y - np.mean(Y)), (Y - np.mean(Y))) 
R_sqr = 1 - (SS_res / SS_tot)
print("R_sqr = ",R_sqr)

#visualize data
plt.scatter(X,Y)
plt.plot(X,y_pred)
plt.show()