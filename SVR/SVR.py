"""
SUPPORT VECTOR REGRESSION
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("folder", help="path to data folder (relative to RandomForest.py)")
parser.add_argument("outfolder",help="output folder name (under ~/RandomForest)")
args = parser.parse_args()


import os
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import itertools

"""
LOAD DATA
"""
folder =  args.folder
outfolder = args.outfolder
#folder = '../simulation_regression/Reg1_M20'
#outfolder = 'Reg1_M20'
# set random seed
np.random.seed(1)
njobs = 2

gene_file = "{}/expression.txt".format(folder)
clinical_file = "{}/clinical.txt".format(folder)
hasclinical = os.path.exists(clinical_file)
resp_file = "{}/response.txt".format(folder)

if hasclinical:
    Z = pd.DataFrame.from_csv(clinical_file)

X = pd.DataFrame.from_csv(gene_file)
Y = pd.DataFrame.from_csv(resp_file)
all_ind = Y.index

if hasclinical:
    X = pd.concat([Z,X],axis=1)

X = X - X.mean()

"""
RUN RANDOM FOREST
"""
C_cand = np.logspace(-1,3,20)
eps_cand = np.linspace(0.1,4,10)
K_cand = ['rbf','poly2','poly3']

#C_cand = np.logspace(-1,2,2)
#eps_cand = np.linspace(0.1,2,2)
#K_cand = ['rbf','poly3']


res = dict()
for i in range(10):
    # split data
    test_lab = "{}/test_label{}.txt".format(folder,i)
    with open(test_lab,'r') as f:
        test_ind = [x.strip() for x in f]
    train_ind = np.setdiff1d(all_ind,np.array(test_ind))
    Xtrain = X.loc[train_ind]
    ytrain = Y.loc[train_ind]
    Xtest = X.loc[test_ind]
    ytest = Y.loc[test_ind]
    # fit data
    print("fitting test data: {}".format(test_lab))
    for c, eps, K in itertools.product(C_cand, eps_cand, K_cand):
        label = "C{:.1f}-eps{:.1f}-{}".format(c,eps,K)
        # fit model
        if K =='rbf':
            model = SVR(kernel = K, epsilon = eps, C = c)
        elif K[:4] == 'poly':
            d = int(K[4])
            model = SVR(kernel = 'poly', epsilon = eps, C = c, degree = d)
        else:
            raise Exception("kernel error")
        model.fit(Xtrain, ytrain.iloc[:,0])
        # prediction
        pred = model.predict(Xtest)
        err = mean_squared_error(ytest.iloc[:,0],pred)
        if label in res:
            res[label].append(err)
        else:
            res[label] = [err]

"""
ORGANIZE OUTPUT AND WRITE TO FILE
"""
out = []
for key,val in res.items():
    out.append( [np.mean(val), np.std(val), val, key] )
out.sort()

if not os.path.exists(outfolder):
    os.makedirs(outfolder)

# write to file
outfile = "{}/results.txt".format(outfolder)
with open(outfile,'w') as f:
    title = "{:20}{:7}{:7}{}".format('parameters','mean','std','detail')
    f.write(title+'\n')
    for x in out:
        m, sd, vals, pars = x
        l = [str(x) for x in np.round(vals,3)]
        msg = "{:20}{:<7}{:<7}{}".format(pars,round(m,3),round(sd,3),' '.join(l))
        f.write(msg+'\n')
    print("results saved to:",outfile)
