"""
utility functions
author: li zeng
"""
import numpy as np
import scipy

"""-----------------
FUNCTIONS
--------------------"""

# initialization function
def init_F_classification(inputs,outputs):
    """
    initialize classification
    """
    ytrain = np.squeeze(inputs.train_response.values)
    N1 = (ytrain == 1).sum()
    N0 = (ytrain ==-1).sum()
    F0 = np.log(N1/N0) # initial value
    if F0==0: F0+=10**(-2)
    outputs.F0 = F0
    # update err,loss
    F_train = np.repeat(F0,inputs.Ntrain) # keep track of F_t(x_i) on training data
    outputs.train_err.append((np.sign(F_train) != ytrain).sum()/len(ytrain))
    outputs.train_loss.append(loss_fun(F_train,ytrain,inputs.problem))
    if not inputs.Ntest == None:
        ytest = np.squeeze(inputs.test_response.values)
        F_test = np.repeat(F0,inputs.Ntest) # keep track of F_t(x_i) on testing data
        outputs.test_err.append((np.sign(F_test) != ytest).sum()/len(ytest))
        outputs.test_loss.append(loss_fun(F_test,ytest,inputs.problem))
    else:
        F_test = None
        ytest = None
    # update trace
    outputs.trace.append([0,np.repeat(0,inputs.Ntrain),0])
    return [F_train,ytrain,F_test,ytest]

def init_F_regression(inputs,outputs):
    """
    initialize regression
    """
    pass

def init_F_survival(inputs,outputs):
    """
    initialize survival
    """
    pass


# loss function
def loss_fun(f,y,problem):
    if problem == "classification":
        return np.mean(np.log(1+np.exp(-y*f)))    
    if problem == "regression":
        pass
    if problem == "survival":
        pass

# line search
def line_search(problem,sharedK,F_train,ytrain,Kdims,pars,sele_loc=None):
    [m,beta,c] = pars    
    
    if sele_loc is None:
        sele_loc = np.array(range(len(ytrain)))
    # get K
    nrow = Kdims[0]
    width = nrow**2
    Km = sharedK[(m*width):((m+1)*width)].reshape((nrow,nrow))
    Km = Km[np.ix_(sele_loc,sele_loc)]
    
    b = Km.dot(beta)+c
    # line search function
    #def temp_fun(x):
    #    return np.log(1+np.exp(-ytrain*(F_train+ x*b))).sum()
    def temp_fun(x):
        return loss_fun(F_train+x*b,ytrain,problem)
    out = scipy.optimize.minimize_scalar(temp_fun)
    if not out.success:
        print("warning: minimization failure")
    return out.x

# sampling 
def subsamp(y,col,fold=3):
    grouped = y.groupby(col)
    out = [list() for i in range(fold)]
    for i in grouped.groups.keys():
        ind = grouped.get_group(i)
        n = ind.shape[0]
        r = list(range(0,n+1,n//fold))
        r[-1] = n+1
        # permute index
        perm_index = np.random.permutation(ind.index)
        for j in range(fold):
            out[j] += list(perm_index[r[j]:r[j+1]])
    return out


"""-----------------
DERIVATIVES
--------------------"""

# first order
def calcu_h(F_train,ytrain,problem):
    if problem == "classification":
        denom  = np.exp(ytrain * F_train) + 1
        return (-ytrain)/denom
    elif problem == "survival":
        pass


# second order
def calcu_q(F_train,ytrain,problem):
    if problem == "classification":
        denom = (np.exp(ytrain * F_train) + 1)**2
        return np.exp(ytrain * F_train)/denom
    elif problem == "survival":
        pass
