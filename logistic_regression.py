import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn import svm
import torch 
import tensorflow


Xbf,y = datasets.load_breast_cancer(return_X_y = True)
one = np.ones((Xbf.shape[0],1))
X = np.concatenate((one, Xbf), axis = 1)

class LogisticRegression():
    def logistic_function(self, t):
        return 1/(1+np.exp(-t))
    
    def cost_function(self, X, y, beta):                 
        z= self.logistic_function(np.dot(X,beta.T))
        return -np.sum(z*np.log(y) + (1-z)*np.log(1-y))
    def grad_function(self, X, y, beta):
        N = X.shape[0]
        z= self.logistic_function(np.dot(X,beta.T))
        return 1/N*np.dot((z-y).T,X)
    def fit(self, X, y, tau=0.1, lr=0.05):        
        loss = []
        beta = np.random.rand(X.shape[1])
                 
        while self.cost_function(X,y,beta) >= tau:  
            loss.append(self.cost_function(X,y,beta))      
            beta -= lr*self.grad_function(X,y,beta)
            
            
            
        self.beta = beta
        self.loss = loss
        return beta
    def predict(self, X,logistic_function):        
        return 
result = LogisticRegression()
print(result.fit(X,y))
