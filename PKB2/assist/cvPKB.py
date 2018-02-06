"""
CV to determine optimal number of iterations
author: li zeng
"""

import assist
from assist.util import line_search, subsamp
import numpy as np
import pandas as pd
from assist.method_L1 import oneiter_L1
from assist.method_L2 import oneiter_L2
from matplotlib import pyplot as plt

"""
CVinputs class
- used to initiate Model class in cross-validation procedure
"""
class CVinputs:
    def __init__(self,inputs,ytrain,ytest):
        self.Ntrain = ytrain.shape[0]
        self.Ntest = ytest.shape[0]
        self.Ngroup = inputs.Ngroup

"""
Cross-Validation function
"""
def CV_PKB(inputs,sharedK,K_train,Kdims,Lambda,nfold=3,ESTOP=30,ncpu=1,parallel=False,gr_sub=False,plot=False):
    ########## split data ###############
    test_inds = subsamp(inputs.train_response,inputs.train_response.columns[0],nfold)
    temp = pd.Series(range(inputs.Ntrain),index= inputs.train_response.index)
    folds = []
    for i in range(nfold):
        folds.append([ temp[test_inds[i]].values, np.setdiff1d(temp.values,temp[test_inds[i]].values)])

    ########## initiate model for each fold ###############
    if inputs.problem == "classification":
        ytrain_ls = [np.squeeze(inputs.train_response.iloc[folds[i][1]].values) for i in range(nfold)]
        ytest_ls = [np.squeeze(inputs.train_response.iloc[folds[i][0]].values) for i in range(nfold)]
        inputs_class = [CVinputs(inputs, ytrain_ls[i], ytest_ls[i]) for i in range(nfold)]
        models = [assist.Classification.PKB_Classification(inputs_class[i], ytrain_ls[i], ytest_ls[i]) for i in range(nfold)]
    elif inputs.problem == 'survival':
        pass
    else:
        pass

    for x in models:
        x.init_F()

    ########## boosting for each fold ###############
    opt_iter = 0
    print([x.test_loss for x in models])
    min_loss = prev_loss =  np.mean( [x.test_loss[0] for x in models] )
    ave_loss = [prev_loss]
    print("-------------------- CV -----------------------")
    print("iteration\tMean test loss")
    for t in range(1,inputs.maxiter+1):
        # one iteration
        for k in range(nfold):
            if inputs.method == 'L2':
                [m,beta,c] = oneiter_L2(sharedK,models[k],Kdims,Lambda=Lambda,ncpu = ncpu,parallel = parallel,\
                sele_loc=folds[k][1])
            if inputs.method == 'L1':
                [m,beta,c] = oneiter_L1(sharedK,models[k],Kdims,Lambda=Lambda,ncpu = ncpu,parallel = parallel,\
                sele_loc=folds[k][1],group_subset = gr_sub)

            # line search
            x = line_search(sharedK,models[k],Kdims,[m,beta,c],sele_loc=folds[k][1])
            beta *= x
            c *= x

            # update model
            cur_Ktrain = K_train[:,:,m][np.ix_(folds[k][1],folds[k][1])]
            cur_Ktest = K_train[:,:,m][np.ix_(folds[k][1],folds[k][0])]
            #print("cur_Ktrain shape: {}\n cur_Ktest shape: {}".format(cur_Ktrain.shape, cur_Ktest.shape))
            models[k].update([m,beta,c],cur_Ktrain,cur_Ktest,inputs.nu)

        # save iteration
        cur_loss = np.mean([x.test_loss[-1] for x in models])
            #update best loss
        if cur_loss < min_loss:
            min_loss = cur_loss
            opt_iter = t
        #ave_err.append(cur_err)
        ave_loss.append(cur_loss)

        # print report
        if t%10 == 0:
            print("%9.0f\t%14.4f" % (t,cur_loss))

        # detect early stop
        if t-opt_iter >= ESTOP:
            print('Early stop criterion satisfied: break CV.')
            print('using iteration number:',opt_iter)
            break
    print("-----------------------------------------------\n")
    # visualization
    if plot:
        folder = inputs.output_folder
        f=plt.figure()
        plt.plot(ave_loss)
        plt.xlabel("iterations")
        plt.ylabel("CV loss")
        f.savefig(folder+'/CV_loss.pdf')
    return opt_iter
