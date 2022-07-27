import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn import svm
import torch 
import tensorflow
boston = datasets.load_boston()
data = boston.data
data = pd.DataFrame(data=data, columns=boston.feature_names)
data['Price'] = boston.target
corr_ft = ['RM','PTRATIO','LSTAT']
Xbf = data[corr_ft]
y = data['Price']
one = np.ones((Xbf.shape[0],1))
X = np.concatenate((one, Xbf), axis = 1)

class LinearRegression():
    def __init__(self,X,y):
        self.X = X
        self.y = y

    def cost_function(self,beta):
        N = X.shape[0]
        return 1/(2*N)*np.linalg.norm(X.dot(beta)-self.y)**2
    def gradient(self,beta):
        N = X.shape[0]
        return 1/N*X.T.dot(X.dot(beta)-self.y)
    def fit(self,tau = 500,gamma = 0.05):
        beta = np.random.rand(X.shape[1])
        N = X.shape[0]
         
        while self.cost_function(beta) >= tau:        
            beta = beta - gamma*self.gradient(beta)
            
        return beta
    def predict(self,X,beta):
        return X.dot(beta)
    def momentum_fit(self,tau = 0.01,gamma = 0.05,alpha = 0.05):
        beta = np.random.rand(X.shape[1])
        old_beta = np.random.rand(X.shape[1])
        N = len(X)
        while self.cost_function(beta) >= tau:
            v_new = old_beta*alpha + self.gradient(beta)
            old_beta = beta
            beta = beta - v_new
        return beta
k = LinearRegression(X,y)
print(k.fit())