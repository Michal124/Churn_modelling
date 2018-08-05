#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 18:02:12 2018

@author: michalgorski
"""

# =============================================================================
#                              Import libraries
# =============================================================================

#Import libraries
import numpy as np
import pandas as pd

# Read data
ds = pd.read_csv("Churn_Modelling.csv")
# Drop unnecessary features

# Split into dependent and independent features
X = ds.iloc[:,3:13].values
y = ds.iloc[:,-1].values

Afeatures = ds.iloc[:,0:13]
Aoutputs = ds.iloc[:,-1]

# ============================================================================= 
#                      Take care of categorical features
# =============================================================================

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Geography
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

# Gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# Delete first column (Dummy variable trap)
X = X[:,1:]


# =============================================================================
#                   Split into training and test set
# =============================================================================

from sklearn.model_selection import train_test_split

X_train , X_test, y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 0 )


print("\n")
print("Before reshaping")
print("\n")
print("Data Shape : \n")
print("X shape : ", X.shape)
print("Train set_x shape : ", X_train.shape)
print("Train set_y shape : ", y_train.shape)
print("Test set_x shape : ", X_test.shape)
print("Test set_y shape : ", y_test.shape)




# =============================================================================
#                           Data Normalization
# =============================================================================

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# =============================================================================
#                          RESHAPING
# =============================================================================


# Reshape cause dependent variables are 1 rank array 

y_train = y_train.reshape(y_train.shape[0],1).T
y_test = y_test.reshape(y_test.shape[0],1).T

X_test = X_test.T
X_train = X_train.T
X = X.T



print("\n")
print("After reshaping")
print("\n")
print("Data Shape : \n")
print("X shape : ", X.shape)
print("Train set_x shape : ", X_train.shape)
print("Train set_y shape : ", y_train.shape)
print("Test set_x shape : ", X_test.shape)
print("Test set_y shape : ", y_test.shape)

#print("Weights shape for argument 3 : ", w.shape)




