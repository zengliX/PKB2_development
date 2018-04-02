"""
L1 penalty method
author: li zeng
"""

import numpy as np
from assist.util import get_K, undefined
import multiprocessing as mp
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)
from sklearn import linear_model

# function to be paralleled
"""
solve L1 penalized regression for pathway m
sharedK: shared kernel matrix
model: model class object
m: pathway index
h: first order derivative
q: second order derivative
sele_loc: index of subset of data to use
"""
def paral_fun_L1(sharedK,Z,model,m,nrow,h,q,Lambda,sele_loc):
    # working Lambda
    new_Lambda= Lambda/2 # due to setting of sklearn.linear_model.Lasso
    # get K
    Km = get_K(sharedK,m,nrow,sele_loc)
    # transform eta, K for penalized regression
    if model.problem in ('classification','survival'):
        eta = model.calcu_eta(h,q)
        w = model.calcu_w(q)
        w_half = model.calcu_w_half(q)
        if model.problem == 'survival' and not model.hasClinical:
            mid_mat = w_half
        else:
            mid_mat = np.eye(len(sele_loc)) - Z.dot( np.linalg.solve(Z.T.dot(w).dot(Z), Z.T.dot(w)) )
        eta_tilde = w_half.dot(mid_mat).dot(eta)
        Km_tilde = w_half.dot(mid_mat).dot(Km)
    elif model.problem == 'regression':
        eta = model.calcu_eta()
        e = np.linalg.solve(Z.T.dot(Z), Z.T)
        mid_mat = np.eye(len(sele_loc)) - Z.dot(e)
        eta_tilde = mid_mat.dot(eta)
        Km_tilde = mid_mat.dot(Km)
    # get beta
    lasso_fit = linear_model.Lasso(alpha = new_Lambda,fit_intercept = False,\
                selection='random',max_iter=20000,tol=10**-4)
    lasso_fit.fit(-Km_tilde,eta_tilde)
    beta = lasso_fit.coef_

    #get gamma
    if model.problem in ('classification','survival'):
        if model.problem == 'survival' and not model.hasClinical:
            gamma = np.array([0.0])
        else:
            gamma = - np.linalg.solve(Z.T.dot(w).dot(Z), Z.T.dot(w)).dot(eta + Km.dot(beta))
    elif model.problem == 'regression':
        gamma = - e.dot(eta + Km.dot(beta))
    # calculate val
    val = np.sum((eta_tilde+Km_tilde.dot(beta))**2)+ Lambda*len(sele_loc)*np.sum(np.abs(beta))
    return [val,[m,beta,gamma]]


"""
find a feasible lambda for L2 problem
K_train: training kernel, shape (Ntrain, Ntrain, Ngroup)
Z: training clinical data, shape (Ntrain, Npred_clin)
model: model class object
Kdims: (Ntrain, Ngroup)
"""
def find_Lambda_L1(K_train,Z,model,Kdims):
    prod = []
    if model.problem in ('classification','survival'):
        h = model.calcu_h()
        q = model.calcu_q()
        eta = model.calcu_eta(h,q)
        w = model.calcu_w(q)
        w_half = model.calcu_w_half(q)
        if model.problem == 'survival' and not model.hasClinical:
            mid_mat = w_half
        else:
            mid_mat = np.eye(Kdims[0]) - Z.dot( np.linalg.solve(Z.T.dot(w).dot(Z), Z.T.dot(w)) )
        eta_tilde = w_half.dot(mid_mat).dot(eta)
    elif model.problem == 'regression':
        eta = model.calcu_eta()
        mid_mat = np.eye(Z.shape[0]) - Z.dot( np.linalg.solve(Z.T.dot(Z), Z.T) )
        eta_tilde = mid_mat.dot(eta)
    for m in range(Kdims[1]):
        Km = K_train[:,:,m]
        Km_tilde = mid_mat.dot(Km)
        prod += list(Km_tilde.dot(eta_tilde))
    return 2*np.percentile(np.abs(prod),85)/Kdims[0]

"""
perform one iteration of L1-boosting
sharedK: kernel matrix
model: model class object
Kdims: (Ntrain, Ngroup)
sele_loc: subset index
group_subset: bool, whether randomly choose a subset of pathways
"""
def oneiter_L1(sharedK,Z,model,Kdims,Lambda,\
               ncpu = 1,parallel=False,sele_loc = None,group_subset = False):
    # whether stochastic gradient boosting
    #print(Kdims)
    if sele_loc is None:
        sele_loc = np.array(range(model.Ntrain))
    if model.problem in ('classification','survival'):
        # calculate derivatives h,q
        h = model.calcu_h()
        q = model.calcu_q()
    elif model.problem == 'regression':
        h = None
        q = None
    # identify best fit K_m
        # random subset of groups
    mlist = range(Kdims[1])
    if group_subset:
        mlist= np.random.choice(mlist,min([Kdims[1]//3,100]),replace=False)
    if parallel:
        pool = mp.Pool(processes =ncpu,maxtasksperchild=300)
        results = [pool.apply_async(paral_fun_L1,args=(sharedK,Z,model,m,Kdims[0],h,q,Lambda,sele_loc)) for m in mlist]
        out = [res.get() for res in results]
        pool.close()
    else:
        out = []
        for m in mlist:
            out.append(paral_fun_L1(sharedK,Z,model,m,Kdims[0],h,q,Lambda,sele_loc))
    return out[np.argmin([x[0] for x in out])][1]
