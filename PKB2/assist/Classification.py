import numpy as np
from assist.Model import BaseModel

class PKB_Classification(BaseModel):
    def __init__(self,inputs,ytrain,ytest):
        super().__init__(inputs,ytrain,ytest)
        # for classification only
        self.problem = 'classification'
        self.train_err = [] # training error
        self.test_err = [] # testing error

    """
    initialize classification
    """
    def init_F(self):
        N1 = (self.ytrain == 1).sum()
        N0 = (self.ytrain ==-1).sum()
        F0 = np.log(N1/N0) # initial value
        if F0==0: F0+=10**(-2)
        self.F0 = F0
        # update training loss, err
        F_train = np.repeat(F0,self.Ntrain)
        self.train_err.append((np.sign(F_train) != self.ytrain).sum()/self.Ntrain)
        self.train_loss.append(self.loss_fun(F_train,self.ytrain))
        # update testing loss, err
        if self.hasTest:
            F_test = np.repeat(F0,self.Ntest)
            self.test_err.append((np.sign(F_test) != self.ytest).sum()/self.Ntest)
            l = self.loss_fun(self.ytest,F_test)
            self.test_loss.append(l)
        else:
            F_test = None
        # update trace
        self.trace.append([0,np.repeat(0,self.Ntrain),0])
        # update F_train, F_test
        self.F_train = F_train
        self.F_test = F_test


    """
    update class after calculation of [m,beta,c] in each iteration
    pars: [m, beta, c]
    K: training kernel matrix
    K1: testing kernel matrix
    rate: learning rate parameter
    """
    def update(self,pars,K,K1,rate):
        super().update(pars,K,K1,rate)
        self.train_err.append((np.sign(self.F_train)!=self.ytrain).sum()/self.Ntrain)
        if self.hasTest:
            self.test_err.append((np.sign(self.F_test)!=self.ytest).sum()/self.Ntest)

    """
    calculate first order derivative
    """
    def calcu_h(self):
        denom  = np.exp(self.ytrain * self.F_train) + 1
        return (-self.ytrain)/denom

    """
    calculate second order derivative
    """
    def calcu_q(self):
        denom = (np.exp(self.ytrain * self.F_train) + 1)**2
        return np.exp(self.ytrain * self.F_train)/denom

    """
    classification loss function, log loss
    """
    def loss_fun(self,y,f):
        return np.mean(np.log(1+np.exp(-y*f)))
