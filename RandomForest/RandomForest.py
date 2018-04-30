"""
RANDOM FOREST FOR REGRESSION
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
njobs = 2

gene_file = "{}/expression.txt".format(folder)
clinical_file = "{}/clinical.txt".format(folder)
resp_file = "{}/response.txt".format(folder)
X = pd.DataFrame.from_csv(gene_file)
Z = pd.DataFrame.from_csv(clinical_file)
Y = pd.DataFrame.from_csv(resp_file)
all_ind = Y.index
X = pd.concat([Z,X],axis=1)

"""
RUN RANDOM FOREST
"""
num_trees = [500,800]
max_depth = [3,5,7,None]

#num_trees = [10,20]
#max_depth = [3,5]

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
    for ntree, M in itertools.product(num_trees, max_depth):
        label = "ntree{}-maxD{}".format(ntree,M)
        rf_fit = ensemble.RandomForestRegressor(n_estimators=ntree,verbose=0,max_depth = M,n_jobs=njobs)
        rf_fit.fit(Xtrain,ytrain.iloc[:,0])
        # prediction error
        pred = rf_fit.predict(Xtest)
        loss = np.mean((ytest.iloc[:,0].values-pred)**2)
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
        l = [str(x) for x in np.round(vals,3)]
        msg = "{:30}{:<10}{:<10}{}".format(pars,round(m,3),round(sd,3),' '.join(l))
        f.write(msg+'\n')
    print("results saved to:",outfile)
