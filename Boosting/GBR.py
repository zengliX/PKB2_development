"""
GRADIENT BOOSTING REGRESSOR
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("folder", help="path to data folder (relative to RandomForest.py)")
parser.add_argument("outfolder",help="output folder name (under ~/RandomForest)")
args = parser.parse_args()


import os
import numpy as np
import pandas as pd
from sklearn import ensemble
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

gene_file = "{}/expression.txt".format(folder)
clinical_file = "{}/clinical.txt".format(folder)
resp_file = "{}/response.txt".format(folder)
X = pd.DataFrame.from_csv(gene_file)
Z = pd.DataFrame.from_csv(clinical_file)
Y = pd.DataFrame.from_csv(resp_file)
all_ind = Y.index
X = pd.concat([Z,X],axis=1)

"""
RUN GRADIENT BOOSTING
"""
maxiter = 800
max_depth = [3,5,7]
rate = [0.002,0.01,0.05]

#maxiter = 20
#max_depth = [3,5]
#rate = [0.01]


res = dict()
for i in range(10):
    # split data
    test_lab = "{}/test_label{}.txt".format(folder,i)
    with open(test_lab,'r') as f:
        test_ind = [x.strip() for x in f]
    train_ind = np.setdiff1d(all_ind,np.array(test_ind))
    ytrain = Y.loc[train_ind]
    Xtrain = X.loc[train_ind]
    Xtest = X.loc[test_ind]
    ytest = Y.loc[test_ind]
    # fit data
    print("fitting test data: {}".format(test_lab))
    for M, r in itertools.product(max_depth, rate):
        label = "maxD{}-rate{}".format(M,r)
        # CV for best iteration
        pars = {'n_estimators':maxiter,'learning_rate':r,'max_depth':M}
        model = ensemble.GradientBoostingRegressor(**pars,subsample=2/3,max_features='sqrt')
        # Fit regressor with out-of-bag estimates
        model.fit(Xtrain, ytrain.iloc[:,0])
        cum_imprv = np.cumsum(model.oob_improvement_)
        opt_iter = np.argmax(cum_imprv)+1
        print("{} optimal iterations: {}".format(label,opt_iter))
        # Fit regressor with optimal maxiter
        pars = {'n_estimators':opt_iter,'learning_rate':r,'max_depth':M}
        model = ensemble.GradientBoostingRegressor(**pars,max_features='sqrt')
        model.fit(Xtrain, ytrain.iloc[:,0])
        pred = model.predict(Xtest)
        loss = np.mean( (pred - ytest.values)**2 )
        if label in res:
            res[label].append(loss)
        else:
            res[label] = [loss]

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
    title = "{:30}{:10}{:10}{}".format('parameters','mean','std','detail')
    f.write(title+'\n')
    for x in out:
        m, sd, vals, pars = x
        msg = "{:30}{:<10}{:<10}{}".format(pars,round(m,3),round(sd,3),np.round(vals,3))
        f.write(msg+'\n')
    print("results saved to:",outfile)
