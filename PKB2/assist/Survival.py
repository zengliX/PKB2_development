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
        self.train_Cind = []
        for i in range(self.Ntrain):
            self.ytrain_tau[i,:] = self.ytrain_time >= self.ytrain_time[i]
        # if test data exists
        if self.hasTest:
            self.ytest_time = self.ytest[:,0]
            self.ytest_cen = self.ytest[:,1]
            self.ytest_delta = 1 - self.ytest[:,1]
            self.ytest_tau = np.zeros((self.Ntest, self.Ntest))
            self.test_Cind = []
            for i in range(self.Ntest):
                self.ytest_tau[i,:] = self.ytest_time >= self.ytest_time[i]

    """
    initialize survival model
    """
    def init_F(self):
        F0 = 0.0
        self.F0 = F0 # initial value
        # update training loss, c-index ...
        F_train = np.repeat(F0, self.Ntrain)
        self.F_train = F_train
        self.train_loss.append(self.loss_fun(self.ytrain, F_train))
        self.train_Cind.append(self.C_index(self.ytrain, self.F_train))
        self.exp_ftrain = np.exp(self.F_train)
        self.exp_indicate = np.dot(self.ytrain_tau, self.exp_ftrain)
        # update testing loss, c-index ...
        if self.hasTest:
            F_test = np.repeat(F0,self.Ntest)
            self.F_test = F_test
            l = self.loss_fun(self.ytest,F_test)
            self.test_loss.append(l)
            self.test_Cind.append(self.C_index(self.ytest, self.F_test))
            exp_ftest = np.exp(F_test)
            self.exp_ftest = exp_ftest
        else:
            self.F_test = None
            self.exp_ftest = None
        # update trace
        self.trace.append([0,np.repeat(0.0,self.Ntrain),np.repeat(0.0,self.Npred_clin)])
        self.fraction_matrix = self.exp_ftrain*np.dot(self.ytrain_tau.T, (self.ytrain_delta/self.exp_indicate))

    """
    calculate first order derivative
    return gradient, shape (Ntrain,)
    """
    def calcu_h(self):
        return -self.ytrain_delta+self.fraction_matrix

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
        self.exp_ftrain = np.exp(self.F_train)
        self.exp_indicate = np.dot(self.ytrain_tau, self.exp_ftrain)
        self.fraction_matrix = self.exp_ftrain*np.dot(self.ytrain_tau.T, (self.ytrain_delta/self.exp_indicate))
        if self.hasTest:
            self.exp_ftest = np.exp(self.F_test)


    def update(self,pars,K,K1,Z,Z1,rate):
        super().update(pars,K,K1,Z,Z1,rate)
        # survival only updates
        self.train_Cind.append(self.C_index(self.ytrain, self.F_train))
        if self.hasTest:
            self.test_Cind.append(self.C_index(self.ytest, self.F_test))
        self.update_att()

    """
    survival loss function, negative log-likelihood
    y: np.array of shape (Ntrain,2)
    f: np.array of shape (Ntrain,)
    """
    def loss_fun(self,y,f):
        N = np.shape(y)[0]
        delta = 1 - y[:,1]
        sur_time = y[:,0]
        tau = np.zeros((N, N))
        for i in range(N):
            tau[i,:] = sur_time >= sur_time[i]
        expy = np.exp(f)
        
        T1 = np.log(np.dot(tau, expy))
        T2 = - delta*(f - T1)
        return np.mean(T2)

    """
    C-index (concordance index) function
    y: np.array of shape (Ntrain,2)
    f: np.array of shape (Ntrain,)
    """
    def C_index(self,y,f):
        ct_pairs = 0.0 # count concordant pairs
        ct=  0 # count total pairs
        for i in range(len(f)-1):
            for j in range(i+1,len(f)):
                val =  self.concordant(y[i,0],y[j,0],y[i,1],y[j,1],f[i],f[j])
                if val is None: continue
                ct_pairs += val
                ct += 1
        return ct_pairs/ct

    def concordant(self,y1,y2,d1,d2,f1,f2):
        """
        y1,y2: survival time
        d1,d2: censoring
        f1,f2: predicted risk
        return None for non-permissible
        """
        # non-permissible
        if (y1<y2 and d1 == 1) or (y2<y1 and d2 == 1):
            return None
        if y1 == y2 and d1==d2==1:
            return None
        # permissible
        if y1 != y2:
            if f1 == f2: return 0.5
            return (1 if (y1>y2) == (f1<f2) else 0)
        else:
            # y1 == y2
            if d1==d2==0:
                return (1 if f1==f2 else 0.5)
            else:
                return (0.5 if f1==f2 else 1 if (d1>d2)==(f2>f1) else 0)

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
