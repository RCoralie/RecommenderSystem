#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Evaluation of the linear SVM algorithm 

Created on Fri Feb 10 11:24:39 2017
@author: Julian
"""

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from numpy import nonzero


class LinearSVM():
    
    def __init__(self, X, y):
        """
            Constructor of the SVM model with paramters : -s 0 -c 4 -B 1
            See liblinear doc to get info for arguments
        """
        
        self._svclassifier = SVC(kernel='linear')  
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.20)  
        
    def train(self):
        """
            Train the SVM model with labels associated to the data
        
            Parameters:
                data -> ndarray, the data to train the model
                labels -> ndarray, labels associted to the data. Must have the same dimenstion
            return:
                SVM model
        
        """
        self._svclassifier.fit(self.X_train, self.y_train)
        #prob  = problem(labels, self._formatData(data))
        #self._model = train(prob, self._param)
        return self._svclassifier
    
    
    def evaluate(self):
        """
            Predict labels from data and compare the result to orginals labels
        
            Parameters:
                data -> ndarray, the data to train the model
                labels -> ndarray, labels associted to the data. Must have the same dimenstion
            return:
                Float, the accuracy 
        """
        
        y_pred = self.predict(self.X_test)
        acc, mse, scc = self.evaluations(self.y_test, y_pred)
        return acc

    def predict(self, data):
        """
            Predict labels from data
        
            Parameters:
                data -> ndarray, the data to train the model
                labels -> ndarray, labels associted to the data. Must have the same dimenstion
            return:
                ndarray, the predicted labels
        """
        
        y_pred = self._svclassifier.predict(data)
        #p_label, p_acc, p_val = predict(labels, self._formatData(data),self._model, '-b 1')
        return y_pred
        
    def evaluations(self, ty, pv):
        """
        evaluations(ty, pv) -> (ACC, MSE, SCC)

        Calculate accuracy, mean squared error and squared correlation coefficient
        using the true values (ty) and predicted values (pv).
        """
        if len(ty) != len(pv):
            raise ValueError("len(ty) must equal to len(pv)")
        total_correct = total_error = 0
        sumv = sumy = sumvv = sumyy = sumvy = 0
        for v, y in zip(pv, ty):
            if y == v: 
                total_correct += 1
            total_error += (v-y)*(v-y)
            sumv += v
            sumy += y
            sumvv += v*v
            sumyy += y*y
            sumvy += v*y 
        l = len(ty)
        ACC = 100.0*total_correct/l
        MSE = total_error/l
        try:
            SCC = ((l*sumvy-sumv*sumy)*(l*sumvy-sumv*sumy))/((l*sumvv-sumv*sumv)*(l*sumyy-sumy*sumy))
        except:
            SCC = float('nan')
        return (ACC, MSE, SCC)

    def _formatData(self, matrix):
        """
            Format matrix2D data into list of dictionary {index:value}
        
            Parameters:
                matrix -> ndarray of data
            return:
                list of dictionary  {indice:value}
        """
        
        if type(matrix) == type([]):
            return matrix
    
        l = []
        for i in range(len(matrix)):
            indVal = {}
            for k in nonzero(matrix[i])[0]:
                indVal[k+1] = matrix[i][k]
            l.append(indVal)
        
        return l