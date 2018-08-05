#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 21:26:50 2018

@author: michalgorski
"""

import numpy as np

def sigmoid(z):
   
    s = 1 / (1 + np.exp(-z))

    return s



def par_init(dim):
   
    w = np.zeros(shape=(dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    return w, b


def propagate(w, b, X, Y):
    
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION 
 
    A = sigmoid(np.dot(w.T, X) + b)  
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))

    # BACKWARD PROPAGATION (TO FIND GRAD)

    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    assert(dw.shape == w.shape)
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost



def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
  

    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation 

        grads, cost = propagate(w, b, X, Y)

        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule 

        w = w - learning_rate * dw  # need to broadcast
        b = b - learning_rate * db

        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
    
    
  
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs




def predict(w, b, X):
    
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))

    A = sigmoid(np.dot(w.T, X) + b)
    
    for i in range(A.shape[1]):
        # Convert probabilities a[0,i] to actual predictions p[0,i]
        if A[0, i] > 0.5 :
            Y_prediction[0, i] = 1 
        else :
            0

    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction




def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
   
    
    # initialize parameters with zeros 
    w, b = par_init(X_train.shape[0])

    # Gradient descent 
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples 
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)


    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

d = model(X_train, y_train, X_test, y_test, num_iterations = 2000, learning_rate = 0.02, print_cost = True)

