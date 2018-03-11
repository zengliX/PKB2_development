from assist.Model import BaseModel
from assist.util import undefined
import scipy
import numpy as np
from scipy.optimize import minimize

class PKB_Regression(BaseModel):
    def __init__(self, inputs, ytrain, ytest):
        super().__init__(inputs, ytrain, ytest)
        # for regression only
        self.inputs = inputs
        self.problem = 'regression'
        self.F_train = []
        self.F_test = []
        self.train_err = [] # training error
        self.test_err = [] # testing error
    """
    initialize regression model
    """
    def init_F(self):
        def tmp_loss(x):
            return(np.mean((self.ytrain - x)**2))
            
        #F0 = scipy.optimize.minimize(tmp_loss, x0 = 0, method = "SLSQP")
        self.F0 = np.mean(self.ytrain)
        #self.F0 = F0.fun
        #print(self.F0)
        self.F_train = np.repeat(self.F0,self.Ntrain)
        if self.hasTest:
            self.F_test = np.repeat(self.F0,self.Ntest)
        else:
            self.F_test = None
        l = tmp_loss(self.F0)
        self.test_loss.append(l)
        

    """
    calculate eta, negative residual
    eta = -r in note
    shape (Ntrain,)
    """
    def calcu_eta(self,Z):
        rt = self.ytrain - self.F_train
        #(np.eye(len(self.ytrain)) - Z.dot(np.linalg.solve(Z.T.dot(Z), Z.T))).dot(rt)
        return rt

    """
    regression loss function, MSE
    y: np.array of shape (Ntrain,)
    f: np.array of shape (Ntrain,)
    """
    def loss_fun(self,y,f):
        return np.mean( (y-f)**2 )
    
    def update(self,pars,K,K1,Z,Z1,rate):
        super().update(pars,K,K1,Z,Z1,rate)
        self.train_err.append((np.sign(self.F_train)!=self.ytrain).sum()/self.Ntrain)
        if self.hasTest:
            self.test_err.append((np.sign(self.F_test)!=self.ytest).sum()/self.Ntest)
        m,beta,gamma = pars
        self.trace.append([m,beta,gamma])
        #=======================================================================
        # print(K.shape)
        # print(K1.shape)
        # print(Z.shape)
        # print(Z1.shape)
        # print(type(K1.T.dot(beta)))
        # print(K1.T.dot(beta).shape)
        # print(K1.T.dot(beta))
        # print(type(Z1.dot(gamma)))
        # print(Z.dot(gamma))
        # print(Z.dot(gamma).shape)
        #=======================================================================
        self.F_train += ( K.dot(beta) + Z.dot(gamma) )*rate
        self.train_loss.append(self.loss_fun(self.ytrain,self.F_train))
        if self.hasTest:
            self.F_test += (K1.T.dot(beta)+ Z1.dot(gamma) )*rate
            new_loss = self.loss_fun(self.ytest,self.F_test)
            self.test_loss.append(new_loss)
