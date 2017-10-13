# -*- coding: utf-8 -*-
"""
main program
author: li zeng
"""


#  ██████  ██████  ███    ███ ███    ███  █████  ███    ██ ██████      ██      ██ ███    ██ ███████
# ██      ██    ██ ████  ████ ████  ████ ██   ██ ████   ██ ██   ██     ██      ██ ████   ██ ██
# ██      ██    ██ ██ ████ ██ ██ ████ ██ ███████ ██ ██  ██ ██   ██     ██      ██ ██ ██  ██ █████
# ██      ██    ██ ██  ██  ██ ██  ██  ██ ██   ██ ██  ██ ██ ██   ██     ██      ██ ██  ██ ██ ██
#  ██████  ██████  ██      ██ ██      ██ ██   ██ ██   ████ ██████      ███████ ██ ██   ████ ███████

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("problem", help="type of analysis (classification/regression/survival)")
parser.add_argument("input", help="Input folder")
parser.add_argument("output", help="Output folder")
parser.add_argument("predictor", help="predictor file")
parser.add_argument("predictor_set",help="file that specifies predictor group structure")
parser.add_argument("response",help="outcome data file")
parser.add_argument("kernel",help="kernel function (rbf/poly3)")
parser.add_argument("method",help="regularization (L1/L2)")
parser.add_argument("-maxiter",help="maximum number of iteration (default 800)")
parser.add_argument("-rate",help="learning rate parameter (default 0.05)")
parser.add_argument("-Lambda",help="penalty parameter")
parser.add_argument("-test",help="file containing test data index")
parser.add_argument("-pen",help="penalty multiplier")
args = parser.parse_args()

# ██ ███    ███ ██████   ██████  ██████  ████████
# ██ ████  ████ ██   ██ ██    ██ ██   ██    ██
# ██ ██ ████ ██ ██████  ██    ██ ██████     ██
# ██ ██  ██  ██ ██      ██    ██ ██   ██    ██
# ██ ██      ██ ██       ██████  ██   ██    ██


from sys import argv
import numpy as np
import sharedmem
#import matplotlib
#matplotlib.use('Agg')
#from matplotlib import pyplot as plt
import assist
from assist.util import loss_fun
from assist.cvPKB import CV_PKB
from assist.method_L1 import oneiter_L1, find_Lambda_L1
from assist.method_L2 import oneiter_L2, find_Lambda_L2
import time
from multiprocessing import cpu_count
#import importlib


#argv = "PKB2.py classification ../data/example ../data/example/new predictor.txt predictor_sets.txt response.txt rbf L1 -maxiter 800 -rate 0.02 -test test_label0.txt -pen 1"
#argv = argv.split(' ')



# ███    ███  █████  ██ ███    ██
# ████  ████ ██   ██ ██ ████   ██
# ██ ████ ██ ███████ ██ ██ ██  ██
# ██  ██  ██ ██   ██ ██ ██  ██ ██
# ██      ██ ██   ██ ██ ██   ████

if __name__ == "__main__":
    #importlib.reload(assist.input_process)
    inputs = assist.input_process.input_obj(args)
    inputs.proc_input()
    # process data
    inputs.data_preprocessing(center=True)
    # split to test, train
    inputs.data_split()
    # report
    inputs.input_summary()
    inputs.model_param()
    print("number of cpus available:",cpu_count())

# ██████  ██    ██ ███    ██      █████  ██       ██████   ██████
# ██   ██ ██    ██ ████   ██     ██   ██ ██      ██       ██    ██
# ██████  ██    ██ ██ ██  ██     ███████ ██      ██   ███ ██    ██
# ██   ██ ██    ██ ██  ██ ██     ██   ██ ██      ██    ██ ██    ██
# ██   ██  ██████  ██   ████     ██   ██ ███████  ██████   ██████

    """---------------------------
    CALCULATE KERNEL
    ----------------------------"""
    #importlib.reload(assist.kernel_calc)
    K_train = assist.kernel_calc.get_kernels(inputs.train_predictors,inputs.train_predictors,inputs)
    if not inputs.Ntest is None:
        K_test= assist.kernel_calc.get_kernels(inputs.train_predictors,inputs.test_predictors,inputs)

    # put K_train in shared memory
    sharedK = sharedmem.empty(K_train.size)
    sharedK[:] = np.transpose(K_train,(2,0,1)).reshape((K_train.size,))
    Kdims = (K_train.shape[0],K_train.shape[2])


    """---------------------------
    initialize outputs object
    ----------------------------"""
    #importlib.reload(assist.outputs)
    outputs = assist.outputs.output_obj(inputs)

    if inputs.problem == "classification":
        F_train, ytrain, F_test, ytest = assist.util.init_F_classification(inputs,outputs)
    elif inputs.problem == "regression":
        pass
    elif inputs.problem == "survival":
        pass
    else:
        print("Analysis ",inputs.problem," not supported"); exit(-1)



# ██████   ██████   ██████  ███████ ████████ ██ ███    ██  ██████
# ██   ██ ██    ██ ██    ██ ██         ██    ██ ████   ██ ██
# ██████  ██    ██ ██    ██ ███████    ██    ██ ██ ██  ██ ██   ███
# ██   ██ ██    ██ ██    ██      ██    ██    ██ ██  ██ ██ ██    ██
# ██████   ██████   ██████  ███████    ██    ██ ██   ████  ██████

    """---------------------------
    BOOSTING PARAMETERS
    ----------------------------"""
    ncpu = min(5,cpu_count())
    Lambda = inputs.Lambda

    # automatic selection of Lambda
    if inputs.method == 'L1' and Lambda is None:
        Lambda = find_Lambda_L1(inputs.problem,K_train,F_train,ytrain,Kdims)
        Lambda *= inputs.pen
        print("L1 method: use Lambda",Lambda)
    if inputs.method == 'L2' and Lambda is None:
        Lambda = find_Lambda_L2(inputs.problem,K_train,F_train,ytrain,Kdims,C=1)
        Lambda *= inputs.pen
        print("L2 method: use Lambda",Lambda)

    # is there a need to run parallel
    if (inputs.Ntrain > 500 or inputs.Ngroup > 40):
        parallel = True
        print("Algorithm: parallel on",ncpu,"cores")
        gr_sub = True
        print("Algorithm: random groups selected in each iteration")
    else:
        parallel = False
        gr_sub = False
        print("Algorithm: parallel algorithm not used")

    ESTOP = 50 # early stop if test_loss have no increase

    """---------------------------
    CV FOR NUMBER OF ITERATIONS
    ----------------------------"""
    opt_iter = CV_PKB(inputs,sharedK,K_train,Kdims,Lambda,nfold=3,ESTOP=ESTOP,\
                      ncpu=1,parallel=parallel,gr_sub=gr_sub,plot=True)

    """---------------------------
    BOOSTING ITERATIONS
    ----------------------------"""

    time0 = time.time()
    print("--------------------- Boosting ------------------------")
    print("iteration\ttrain loss\ttest loss\t    time")
    for t in range(1,opt_iter+1):
        # one iteration
        if inputs.method == 'L2':
            [m,beta,c] = oneiter_L2(inputs.problem,sharedK,F_train,ytrain,Kdims,\
                    Lambda=Lambda,ncpu = ncpu,parallel = parallel,\
                    sele_loc=None,group_subset = gr_sub)
        if inputs.method == 'L1':
            [m,beta,c] = oneiter_L1(inputs.problem,sharedK,F_train,ytrain,Kdims,\
                    Lambda=Lambda,ncpu = ncpu,parallel = parallel,\
                    sele_loc=None,group_subset = gr_sub)

        # line search
        x = assist.util.line_search(inputs.problem,sharedK,F_train,ytrain,Kdims,[m,beta,c])
        beta *= x
        c *= x

        # update outputs
        outputs.trace.append([m,beta,c])
        F_train += (K_train[:,:,m].dot(beta) + c)*inputs.nu
        outputs.train_loss.append(loss_fun(F_train,ytrain,inputs.problem))

        if inputs.problem == "classification": # only classification has err
            outputs.train_err.append((np.sign(F_train)!=ytrain).sum()/len(ytrain))
        if inputs.Ntest>0:
            F_test += (K_test[:,:,m].T.dot(beta)+ c)*inputs.nu
            new_loss = loss_fun(F_test,ytest,inputs.problem)
            outputs.test_loss.append(loss_fun(F_test,ytest,inputs.problem))
            if inputs.problem == "classification":
                outputs.test_err.append((np.sign(F_test)!=ytest).sum()/len(ytest))

        # print time report
        if t%10 == 0:
            iter_persec = t/(time.time() - time0) # time of one iteration
            rem_time = (opt_iter-t)/iter_persec # remaining time
            print("%9.0f\t%10.4f\t%9.4f\t%8.4f" % \
                  (t,outputs.train_loss[t],outputs.test_loss[t],rem_time/60))
    print("-------------------------------------------------------")



# ██████  ███████ ███████ ██    ██ ██   ████████ ███████
# ██   ██ ██      ██      ██    ██ ██      ██    ██
# ██████  █████   ███████ ██    ██ ██      ██    ███████
# ██   ██ ██           ██ ██    ██ ██      ██         ██
# ██   ██ ███████ ███████  ██████  ███████ ██    ███████

    ## show results
    # trace
    if inputs.problem == "classification":
        f = outputs.show_err()
        f.savefig(inputs.output_folder + "/err.pdf")
    f = outputs.show_loss()
    f.savefig(inputs.output_folder + "/loss.pdf")

    # opt weights
    [weights,f0] = outputs.group_weights(opt_iter-1,plot=True)
    f0.savefig(inputs.output_folder + "/opt_weights.pdf")

    # weights paths
    [path_mat,f1] = outputs.weights_path(plot=True)
    f1.savefig(inputs.output_folder + "/weights_path.pdf")


# ███████  █████  ██    ██ ███████
# ██      ██   ██ ██    ██ ██
# ███████ ███████ ██    ██ █████
#      ██ ██   ██  ██  ██  ██
# ███████ ██   ██   ████   ███████

    # save outputs to files
    import pickle
    out_file = inputs.output_folder + "/results.pckl"
    f = open(out_file,'wb')
    outputs.inputs.test_predictors =None
    outputs.inputs.train_predictors = None
    outputs.inputs.test_response=None
    outputs.inputs.train_response=None
    outputs.inputs.pred_sets=None
    pickle.dump(outputs,f)
    f.close()
    print("results saved to:",out_file)
