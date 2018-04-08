from assist.Model import BaseModel
from assist.util import undefined
from assist.util import testInf
import numpy as np
import numpy.linalg as npl
# import lifelines


# censor = 1 means censor

class PKB_Survival(BaseModel):
    def __init__(self,inputs,ytrain,ytest):
        super().__init__(inputs,ytrain,ytest)
        # for survival only
        self.problem = 'survival'
        self.ytrain_time = self.ytrain[:,0]
        self.ytrain_cen = self.ytrain[:,1]
        temp = 1 - self.ytrain[:,1]
        self.ytrain_delta = temp
        self.ytrain_tau = np.zeros((self.Ntrain, self.Ntrain))
        for i in range(self.Ntrain):
            self.ytrain_tau[i,:] = self.ytrain_time >= self.ytrain_time[i]
        if self.hasTest:
            self.ytest_time = self.ytest[:,0]
            self.ytest_cen = self.ytest[:,1]
            self.ytest_delta = np.squeeze(1 - self.ytest[:,1])
            self.ytest_tau = np.zeros((self.Ntest, self.Ntest))
            for i in range(self.Ntest):
                self.ytest_tau[i,:] = self.ytest_time >= self.ytest_time[i]

    """
    initialize survival model
    """
    def init_F(self):
        F0 = 0.0
        self.F0 = F0 # initial value
        # update training loss, err
        F_train = np.repeat(F0, self.Ntrain)
        self.train_loss.append(self.loss_fun(self.ytrain, F_train))
        # update testing loss, err
        if self.hasTest:
            F_test = np.repeat(F0,self.Ntest)
            l = self.loss_fun(self.ytest,F_test)
            self.test_loss.append(l)
            exp_ftest = np.squeeze(np.exp(F_test))
        else:
            F_test = None
            exp_ftest = None
        # update trace
        self.trace.append([0,np.repeat(0.0,self.Ntrain),np.repeat(0.0,self.Npred_clin)])
        # update F_train, F_test
        self.F_train = F_train
        self.F_test = F_test
        self.exp_ftrain = np.squeeze(np.exp(self.F_train))
        self.exp_indicate = np.squeeze(np.dot(self.ytrain_tau, self.exp_ftrain))
        self.fraction_matrix = np.squeeze(self.exp_ftrain*np.dot(self.ytrain_tau.T, (self.ytrain_delta/self.exp_indicate)))
        self.exp_ftest = np.squeeze(exp_ftest)

    """
    calculate first order derivative
    return gradient, shape (Ntrain,)
    """
    def calcu_h(self):
        return np.squeeze(-self.ytrain_delta+self.fraction_matrix)

    """
    calculate second order derivative
    return hessian matrix, shape (Ntrain, Ntrain)
    """
    def calcu_q(self):
        temp = self.ytrain_delta/self.exp_indicate*self.ytrain_tau.T
        mat = np.matrix(self.exp_ftrain)
        Q = np.diag(self.fraction_matrix) - np.multiply(np.dot(mat.T, mat),np.dot(temp,temp.T))
        return np.array(Q)

    def update_att(self):
        self.exp_ftrain = np.squeeze(np.exp(self.F_train))
        self.exp_indicate = np.squeeze(np.dot(self.ytrain_tau, self.exp_ftrain))
        self.fraction_matrix = np.squeeze(self.exp_ftrain*np.dot(self.ytrain_tau.T, (self.ytrain_delta/self.exp_indicate)))
        if self.hasTest:
            self.exp_ftest = np.squeeze(np.exp(self.F_test))


    def update(self,pars,K,K1,Z,Z1,rate):
        super().update(pars,K,K1,Z,Z1,rate)
        self.update_att()
    """
    survival loss function, negative log-likelihood
    y: np.array of shape (Ntrain,2)
    f: np.array of shape (Ntrain,)
    """
    def loss_fun(self,y,f):
        N = np.shape(y)[0]
        delta = 1 - y[:,1]
        train_time = y[:,0]
        def calc_de(j):
            E1 = np.exp(f)
            E2 = train_time >= train_time[j]
            E = E1*E2
            return E.sum()
        def calc_de_array():
            S = np.repeat(0.0, N)
            for i in range(N):
                S[i] = calc_de(i)
            return(S)
        T1 = np.log(calc_de_array())
        T2 = - delta*(f - T1)
        return np.mean(T2)

    """
    calculate eta，W, W^(1/2) from h and q
    h: gradient, shape (Ntrain,)
    q: Hessian, shape (Ntrain, Ntrain)
    """
    def calcu_eta(self,h,q):
        u, s, vh = npl.svd(q)
        abss = np.abs(s)
        med = np.median(abss)
        s = s[abss>=0.0001*med]
        u = u[:,abss>=0.0001*med]
        vh = vh[abss>=0.0001*med,:]
        S = np.diag(1/s)
        return np.dot(np.dot(u, np.dot(S, vh)), h)

    def calcu_w(self,q):
        return q/2

    def calcu_w_half(self,q):
        u, s, vh = npl.svd(q/2)
        S = np.diag(np.sqrt(s))
        return np.dot(u, np.dot(S, vh))
