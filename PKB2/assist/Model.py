"""
Base model class
author: li zeng
"""

import numpy as np

"""
Model class is used to
- initialize parameters
- keep track of each iteration in boosting
- at the end of program, will be used for visualization
- only the outcome (ytrain, ytest) and predicting function (Ftrain, Ftest) will be stored
- other parameters kernel matrix, Z matrix ... will be passed in as function arguments
"""
class BaseModel:
    """
    model initialization
    coef_mat: beta coefficients for pathways, shape (Ntrain, Ngroup)
    trace: increment parameters in each boosting iteration
    train_loss: training loss of each iteration
    test_loss: testing loss of each iteration
    hasTest: bool, indicating existence of test data
    """
    def __init__(self,inputs,ytrain,ytest):
        self.ytrain = ytrain
        self.ytest = ytest
        self.Ntrain = inputs.Ntrain
        self.Ntest = inputs.Ntest
        self.hasTest = not inputs.Ntest is None
        self.F0 = None

        # tracking of performance
        self.coef_mat =  np.zeros([inputs.Ntrain,inputs.Ngroup])
        self.trace = []  # keep track of each iteration
        self.train_loss = [] # loss function at each iteration
        self.test_loss = []

    """
    initialize F_train, ytrain, F_test, ytest
    """
    def init_F(self):
        self.F_train = []
        self.F_test = []
        self.ytrain = None
        self.ytest = None

    """
    update [F_train, F_test, trian_loss, test_loss] after calculation of [m,beta,c] in each iteration
    pars: [m, beta, c]
    K: training kernel matrix
    K1: testing kernel matrix
    rate: learning rate parameter
    """
    def update(self,pars,K,K1,rate):
        m,beta,c = pars
        self.trace.append([m,beta,c])
        self.F_train += (K.dot(beta) + c)*rate
        self.train_loss.append(self.loss_fun(self.ytrain,self.F_train))
        if self.hasTest:
            self.F_test += (K1.T.dot(beta)+ c)*rate
            new_loss = self.loss_fun(self.ytest,self.F_test)
            self.test_loss.append(new_loss)


    """------------------------------------
    other functions to be added in child classes

    #first order derivative
    def calcu_q(self):
        pass

    #second order derivative
    def calcu_h(self):
        pass

    #loss function
    def loss_fun(self,y,f):
        pass
    ---------------------------------------"""
