#!/usr/bin/env python3
"""Defines a neural network"""

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """
    Class defines a neural network

    Args:
        nx: number of input features
        nodes: number of nodes found in the hidden layer
        W1: Weight vector of the hidden layer
        W2: Weight vector of the output neuron
        b1: Bias of the hidden layer
        b2: Bias of the output layer
        A1: Prediction of the hidden layer
        A2: Prediction of the output layer
    """
    def __init__(self, nx, nodes):
        """
        class constructor
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')

        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')

        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.nx = nx
        self.nodes = nodes
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        '''
            Getter
        '''
        return self.__W1

    @property
    def b1(self):
        '''
            Getter
        '''
        return self.__b1

    @property
    def A1(self):
        '''
            Getter
        '''
        return self.__A1

    @property
    def W2(self):
        '''
            Getter
        '''
        return self.__W2

    @property
    def b2(self):
        '''
            Getter
        '''
        return self.__b2

    @property
    def A2(self):
        '''
            Getter
        '''
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        """
        self.__A1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-self.__A1))
        self.__A2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-self.__A2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        cost = -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))) / m
        return cost

    def evaluate(self, X, Y):
        """
        evaluates the neural network's prediction
        """
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        prediction = np.where(A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        """
        m = Y.shape[1]
        dz2 = A2 - Y
        dw2 = (1 / m) * np.matmul(dz2, A1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.matmul(self.__W2.T, dz2) * (A1 * (1 - A1))
        dw1 = (1 / m) * np.matmul(dz1, X.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
        self.__W2 = self.__W2 - (alpha * dw2)
        self.__b2 = self.__b2 - (alpha * db2)
        self.__W1 = self.__W1 - (alpha * dw1)
        self.__b1 = self.__b1 - (alpha * db1)
        return self.__W1, self.__b1, self.__W2, self.__b2

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
            Trains the neural network using gradient descent
        """
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')

        if iterations < 1:
            raise ValueError('iterations must be a positive integer')

        if type(alpha) is not float:
            raise TypeError('alpha must be a float')

        if alpha < 0:
            raise ValueError('alpha must be positive')

        if graph or verbose:
            if type(step) is not int:
                raise TypeError('step must be an integer')

            if step < 1 or step > iterations:
                raise ValueError('step must be positive and <= iterations')

        costs = []
        iteration_list = []

        for i in range(iterations+1):
            # Forward propagation
            A1, A2 = self.forward_prop(X)

            # compute the cost
            cost = self.cost(Y, A2)
            costs.append(cost)
            iteration_list.append(i)

            if verbose and i % step == 0:
                print("Cost after iteration {}: {}".format(i, cost))

            # gradient descent for the
            self.gradient_descent(X, Y, A1, A2)

        # Evaluation after training
        evaluation = self.evaluate(X, Y)

        # plot the cost function
        if graph:
            plt.plot(iteration_list, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return evaluation
