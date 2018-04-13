"""
python MKLpy package implementation
author: li zeng
"""

#########################################
######### PREPARATION ##################
#################################### 

import os
import numpy as np
import pandas as pd
import yaml
from sys import argv
import pickle
from sklearn import metrics
import sys
from sklearn.preprocessing import scale


#########################################
######### INPUT ##################
#################################### 

# set random seed
np.random.seed(1)
njobs = 2


# commandline inputs
config_file = argv[1]
outfolder = argv[2]
KN = argv[3]

print("input file:",config_file)
print("kernel:",KN)
print()

f = open(config_file)
info =yaml.load(f)
f.close()
X = pd.DataFrame.from_csv(info['input_folder']+'/'+info['predictor'])
Y = pd.DataFrame.from_csv(info['input_folder']+'/'+info['response'])
pred_set = pd.Series.from_csv(info['input_folder']+'/'+info['predictor_set'])

# center X
scale(X,copy=False,with_std=False)

# calculate kernels
def get_kernels(X,Y,pred_set,kernel):
    out =[]
    
    drop_ind = []
    for ind in pred_set.index:
        genes = pred_set[ind].split(" ")
        shared = np.intersect1d(X.columns.values,genes)
        if len(shared)==0: 
            drop_ind.append(ind)
            continue
        a = X[shared]
        b = Y[shared]
        if kernel=='rbf':
            out.append( metrics.pairwise_kernels(a,b,metric='rbf'))
        elif kernel[:4]=='poly':
            out.append ( metrics.pairwise.polynomial_kernel(a,b,degree=3,gamma=1/len(shared)))
        else:
            print("wrong kernel option: "+kernel+'\n')
            sys.exit(-3)
    
    for ind in drop_ind:
        pred_set.drop(ind)
        print('dropping group: ',ind)
    return out


K_all = get_kernels(X,X,pred_set,kernel='poly3')


#########################################
######### MKL ##################
#################################### 

from MKLpy.algorithms import EasyMKL

out_obs = []
for i in range(3):
    test_lab = info['input_folder']+'/test_label'+str(i)+'.txt'
    f  = open(test_lab,'r')
    test_ind = [x.strip() for x in f]
    f.close()
    train_ind = np.setdiff1d(X.index.values,np.array(test_ind))    
    
    y_train = Y.loc[train_ind]
    y_test = Y.loc[test_ind]
    K_train = get_kernels(X.loc[train_ind],X.loc[train_ind],pred_set,KN)
    K_test = get_kernels(X.loc[train_ind],X.loc[test_ind],pred_set,KN)
    K_test = [x.T for x in K_test]
    
    clf = EasyMKL(verbose = True,kernel = 'precomputed')
    clf = clf.fit(K_train,y_train.values)     
    # clf.get_params()
    # prediction error
    err= 1 - clf.score(K_test,y_test.values)
    w = clf.weights
    out_obs.append([err,clf.weights,pred_set.index])    


##########################################
######### SAVE ###########################
#######################################
if not os.path.exists(outfolder):
    os.makedirs(outfolder)

out_file = outfolder + "/results.pckl"
f = open(out_file,'wb')
pickle.dump(out_obs,f)
f.close()
print("results saved to:",out_file)

f = open(outfolder+'/report.txt','w')
f.write("prediction error: "+str([x[0] for x in out_obs]) )
f.close()
