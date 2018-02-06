# -*- coding: utf-8 -*-
"""
The outputs class
author: li zeng
"""

import numpy as np
from matplotlib import pyplot as plt

def weight_calc(mat):
    weights = []
    for j in range(mat.shape[1]):
        weights.append( np.sqrt((mat[:,j]**2).sum()) )
    return weights


"""
output object:
- with model as a member
- has visualization function to interact with model

"""
class output_obj:
    # initialization
    def __init__(self,model,inputs):
        self.inputs = inputs
        self.model = model
        return

    """
    show the trace of fitting
    """
    def show_err(self):
        f = plt.figure()
        plt.plot(self.model.train_err,'b')
        plt.text(len(self.model.train_err),self.model.train_err[-1], "training error")
        if self.model.hasTest:
            plt.plot(self.model.test_err,'r')
            plt.text(len(self.model.test_err),self.model.test_err[-1], "testing error")
        plt.xlabel("iterations")
        plt.ylabel("classification error")
        plt.title("Classifiction errors in each iteration")
        return f

    """
    show the trace plot of loss function
    """
    def show_loss(self):
        f = plt.figure()
        plt.plot(self.model.train_loss,'b')
        plt.text(len(self.model.train_loss),self.model.train_loss[-1], "training loss")
        if self.model.hasTest:
            plt.plot(self.model.test_loss,'r')
            plt.text(len(self.model.test_loss),self.model.test_loss[-1], "testing loss")
        plt.xlabel("iterations")
        plt.ylabel("loss function")
        plt.title("Loss function at each iteration")
        return f

    """
    return group weights at iteration t
    """
    def group_weights(self,t,plot=True):
        self.model.coef_mat.fill(0)
        # calculate coefficient matrix at step t
        for i in range(t+1):
            [m,beta,c] = self.model.trace[i]
            self.model.coef_mat[:,m] += beta*self.inputs.nu

        # calculate pathway weights
        weights = weight_calc(self.model.coef_mat)

        # visualization
        if plot:
            f=plt.figure()
            plt.bar(range(1,self.inputs.Ngroup+1),weights)
            plt.xlabel("groups")
            plt.ylabel("group weights")

        return [weights,f]

    """
    show the path of weights for each group
    """
    def weights_path(self,plot=True):
        self.model.coef_mat.fill(0)
        # calculate coefficient matrix at step t
        weight_mat = np.zeros([len(self.model.train_err),self.inputs.Ngroup])

        # calculate weights for each iteration
        for i in range(1,len(self.model.train_err)):
            [m,beta,c] = self.model.trace[i]
            self.model.coef_mat[:,m] += beta*self.inputs.nu
            weight_mat[i,:] = weight_mat[i-1,:]
            weight_mat[i,m] =  np.sqrt((self.model.coef_mat[:,m]**2).sum())

        # weights at opt_t
        first5 = weight_mat[-1,:].argsort()[-5:][::-1]

        # visualization
        if plot:
            f1=plt.figure()
            for m in range(weight_mat.shape[1]):
                if m in first5:
                    plt.plot(weight_mat[:,m],label=str(self.inputs.group_names[m]))
                else:
                    plt.plot(weight_mat[:,m])
            plt.legend()
            plt.xlabel("iterations")
            plt.ylabel("weights")
            plt.title("group weights dynamics")
        return [weight_mat,f1]

    """
    clean up big data information before being pickled
    """
    def clean_up(self):
        self.inputs.test_predictors =None
        self.inputs.train_predictors = None
        self.inputs.test_response=None
        self.inputs.train_response=None
        self.inputs.pred_sets=None
