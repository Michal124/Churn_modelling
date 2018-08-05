#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 19:52:26 2018

@author: michalgorski
"""

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import seaborn as sns
sns.set()

import numpy as np

# =============================================================================
#                     PLOT ACTIVATION FUNCTION
# =============================================================================


x1 = np.linspace(-8,8)
x2 = np.array([np.linspace(-8,8,6)])


plt.figure(1)
plt.plot(x1,sigmoid(x1),"r")
plt.scatter(x2,sigmoid(x2))
pylab.text(4, 0.8, r'$\sigma(x)=\frac{1}{1+e^{-x}}$', fontsize=15)
plt.title("Sigmoid function")
plt.xlabel("X axis")
plt.ylabel("Y axis")

plt.show()

del x1,x2


# =============================================================================
#                       PLOT CONFUSION MATRIX
# =============================================================================

from sklearn.metrics import confusion_matrix
import seaborn as sns

prediction_test = d["Y_prediction_test"]
prediction1 = prediction_test.T
y_test1 = y_test.T 
cnf_matrix = confusion_matrix(y_test1,prediction1)



plt.figure(2,figsize=(10,5))
plt.subplot(2,1,1)
plt.title("Confusion Matrix")

# Confusion Matrix
sns.heatmap(cnf_matrix,annot=True, fmt='.0f',robust=True,
            cmap = "Blues", linewidths=1, linecolor='black',xticklabels=["stay","bye"], yticklabels=["stay","bye"])


# Normalized Confusion Matrix
plt.subplot(2,1,2)
plt.title("Normalized Confusion Matrix")
cnf_matrix_normalized = cnf_matrix/cnf_matrix.sum(axis=0)
sns.heatmap(cnf_matrix_normalized ,annot=True, fmt='.2f', 
            cmap = "Blues", linewidths=1, linecolor='black',xticklabels=["stay","bye"], yticklabels=["stay","bye"])

del y_test1, prediction1


# =============================================================================
#                        PLOT LEARNING CURVE
# =============================================================================

plt.figure(3)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

# =============================================================================
#                        PLOT LEARNING RATES TEST
# =============================================================================

plt.figure(4)
learning_rates = [0.5,0.02,0.01,0.005, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(X_train, y_train, X_test, y_test, num_iterations = 2000, learning_rate = i, print_cost = True)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))


legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')

plt.ylabel('cost')
plt.xlabel('iterations')
plt.show()