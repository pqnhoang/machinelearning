from statistics import mean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn import svm
import torch 
import tensorflow


cancer = datasets.load_breast_cancer()
data = cancer.data
data = pd.DataFrame(data=data, columns=cancer.feature_names)
test =['worst fractal dimension','mean radius']
Xbf = data[test]
y = cancer.target
one = np.ones((Xbf.shape[0],1))
X = np.concatenate((one, Xbf), axis = 1)

class LogisticRegression():
    def logistic_function(self, t):
        return 1/(1+np.exp(-t))
    
    def cost_function(self, X, y, beta):                 
        z= self.logistic_function(np.dot(X,beta.T))
        return -np.sum(y*np.log(z)+(1-y)*np.log(1-z))
    def grad_function(self, X, y, beta):
        N = X.shape[0]
        z= self.logistic_function(np.dot(X,beta.T))
        return 1/N*np.dot((z-y).T,X)
        '''
    def fit(self, X, y, tau=5, lr=0.02):        
        loss = []
        beta = np.random.rand(X.shape[1])
                 
        while self.cost_function(X,y,beta) >= tau:  
            loss.append(self.cost_function(X,y,beta))      
            beta -= lr*self.grad_function(X,y,beta)
            
            
            
        self.beta = beta
        self.loss = loss
        return beta
    '''
    def fit(self,X,y,iteration = 100000,lr = 0.001):
        loss = []
        beta = np.random.rand(X.shape[1])
                 
        for i in range(iteration):
            loss.append(self.cost_function(X,y,beta))      
            beta -= lr*self.grad_function(X,y,beta)
        self.beta = beta
        self.loss = loss
        return beta
    def predict(self, X):        
        return self.logistic_function(np.dot(X,self.beta.T))
result = LogisticRegression()
print(result.fit(X,y))
print(result.predict(X))
print(y)
