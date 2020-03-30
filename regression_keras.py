import numpy as np
import pandas as pd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from util import get_regression_data

from variables import *

class Regression(object):
    def __init__(self):
        X, Y = get_regression_data()
        self.X = X
        self.Y = Y

    def neural_network(self):
        model = Sequential()
        model.add(Dense(1, input_shape=(1,)))
        model.compile(
            optimizer=SGD(lr,momentum),
            loss='mse')
        model.fit(self.X, self.Y, epochs=num_epochs)
        self.model= model

    # def log_polynomial_regression(self):


if __name__ == "__main__":
    def plot_data(Xs, Ys, log_y=False, show_fig=True):
        plt.scatter(Xs, Ys, color='red')
        title_ = 'moore law' if not log_y else 'moore law logrithm'
        img = 'moore law.png' if not log_y else 'moore law logrithm.png'

        plt.title(title_)
        plt.ylabel('transistor count')
        plt.xlabel('year')
        plt.savefig(img)
        plt.show(img)

    def get_regression_data():
        df = pd.read_csv(csv_path, header=None)
        Xs = df.iloc[:,0].to_numpy().reshape(-1,1)

        Ys = df.iloc[:,1].to_numpy()

        plot_data(Xs, Ys)
        Ys = np.log(Ys)
        plot_data(Xs, Ys,True)

        Xs = Xs - Xs.mean()
        return Xs,Ys
        
    reg = Regression()
    reg.neural_network()
