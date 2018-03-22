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
        self.ytest_time = self.ytest[:,0]
        self.ytest_cen = self.ytest[:,1]

    """
    initialize survival model
    """
    def init_F(self):
        F0 = 100.0
        self.F0 = F0 # initial value
        # update training loss, err
        F_train = np.repeat(F0, self.Ntrain)
        self.train_loss.append(self.loss_fun(self.ytrain, F_train))
        # update testing loss, err
        if self.hasTest:
            F_test = np.repeat(F0,self.Ntest)
            l = self.loss_fun(self.ytest,F_test)
            self.test_loss.append(l)
        else:
            F_test = None
        # update trace
        self.trace.append([0,np.repeat(0,self.Ntrain),np.append( np.repeat(0,self.Npred_clin),[F0] )])
        # update F_train, F_test
        self.F_train = F_train
        self.F_test = F_test


    """
    a function to calculate a temp value
    """
    def calcu_denom(self, j):
        E1 = np.exp(self.F_train)
        E2 = self.ytrain_time >= self.ytrain_time[j]
        E = E1*E2
        return E.sum()
    
    def calcu_denom_array(self):
        N = self.Ntrain
        S = np.zeros(N)
        for i in range(N):
            S[i] = self.calcu_denom(i)
        return(S)


    """
    calculate first order derivative
    return gradient, shape (Ntrain,)
    """
    def calcu_h(self):
        delta = 1 - self.ytrain_cen
        E1 = np.exp(self.F_train)
        N2 = np.repeat(0, self.Ntrain)
        temp = self.calcu_denom_array()
        for k in range(self.Ntrain):
            E2 = delta*(self.ytrain_time[k] >= self.ytrain_time)/temp
            S1 = E2.sum()
            N2[k] = S1
        return - delta + E1 * N2

    """
    calculate second order derivative
    return hessian matrix, shape (Ntrain, Ntrain)
    """
    def calcu_q(self):
        delta = 1 - self.ytrain_cen
        Q = np.zeros((self.Ntrain, self.Ntrain))
        E1 = np.exp(self.F_train)
        sqE1 = E1**2
        temp = self.calcu_denom_array()
        for i in range(self.Ntrain):
            for l in range(self.Ntrain):
                if i == l:
                    denomj = delta*(self.ytrain_time[i]>=self.ytrain_time)/temp
                    sqdenomj = delta*(self.ytrain_time[i]>=self.ytrain_time)/temp**2
                    sumdenomj = denomj.sum()
                    sumsqdenomj = sqdenomj.sum()
                    Q[i,i] = E1[i]*sumdenomj - sumsqdenomj*sqE1[i]
                elif i != l:
                    denom = delta*(self.ytrain_time[i]>=self.ytrain_time)*(self.ytrain_time[l]>=self.ytrain_time)/temp**2
                    sumdenom = denom.sum()
                    Q[i,l] = - sumdenom * E1[i]*E1[l]
        return Q


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
            S = np.zeros(N)
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
        s = s[abss>=0.01*med]
        u = u[:,abss>=0.01*med]
        vh = vh[abss>=0.01*med,:]
        S = np.diag(1/s)
        return np.dot(np.dot(u, np.dot(S, vh)), h)

    def calcu_w(self,q):
        return q/2

    def calcu_w_half(self,q):
        u, s, vh = npl.svd(q/2)
        S = np.diag(np.sqrt(s))
        return np.dot(u, np.dot(S, vh))
