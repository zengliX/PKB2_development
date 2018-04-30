"""
REGRESSION WITH LASSO, RIDGE, ELASTICNET
"""

"""
RANDOM FOREST FOR REGRESSION
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("folder", help="path to data folder (relative to RandomForest.py)")
parser.add_argument("outfolder",help="output folder name (under ~/RandomForest)")
parser.add_argument("-c", help="use clinical data only")
args = parser.parse_args()


import os
import numpy as np
import pandas as pd
import sklearn.linear_model as LM
from sklearn.metrics import mean_squared_error

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

Z = pd.DataFrame.from_csv(clinical_file)
Y = pd.DataFrame.from_csv(resp_file)
if args.c is None:
    X = pd.DataFrame.from_csv(gene_file)
    X = pd.concat([Z,X],axis=1)
else:
    X = Z
all_ind = Y.index
print("X shape: {}".format(X.shape))


"""
RUN LINEAR MODELS
"""
res_lasso = []
res_ridge = []
res_enet = []

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
    print("fitting test data: {}".format(test_lab))

    """
    LASSO FIT
    """
    lasso_cv = LM.LassoCV(n_alphas=100, random_state=0, cv = 3)
    lasso_cv.fit(Xtrain,ytrain.iloc[:,0])
    sele_alpha = lasso_cv.alpha_
    lasso = LM.Lasso(alpha= sele_alpha, random_state=0)
    lasso.fit(Xtrain,ytrain)
    pred = lasso.predict(Xtest)
    err = mean_squared_error(ytest.iloc[:,0],pred)
    res_lasso.append(err)
    print("LASSO par {}, error {}".format(sele_alpha,err))

    """
    RIDGE FIT
    """
    alphas = np.logspace(-1,4,num=100)
    ridge_cv = LM.RidgeCV(alphas, cv = 3)
    ridge_cv.fit(Xtrain,ytrain.iloc[:,0])
    sele_alpha = ridge_cv.alpha_
    ridge = LM.Ridge(alpha = sele_alpha)
    ridge.fit(Xtrain,ytrain)
    pred = ridge.predict(Xtest)
    err = mean_squared_error(ytest.iloc[:,0],pred)
    res_ridge.append(err)
    print("Ridge par {},error {}".format(sele_alpha,err))

    """
    ELASTICNET FIT
    """
    l1_ratios = np.linspace(0.1,1,10)
    Enet_cv = LM.ElasticNetCV(l1_ratios, n_alphas = 30, cv = 3)
    Enet_cv.fit(Xtrain,ytrain.iloc[:,0])
    sele_ratio = Enet_cv.l1_ratio_
    sele_alpha = Enet_cv.alpha_
    Enet = LM.ElasticNet(alpha = sele_alpha, l1_ratio = sele_ratio)
    Enet.fit(Xtrain, ytrain.iloc[:,0])
    pred = Enet.predict(Xtest)
    err = mean_squared_error(ytest, pred)
    res_enet.append(err)
    print("ElasticNet ratio {}, alpha {},error {}".format(sele_ratio, sele_alpha,err))
    print()


"""
ORGANIZE OUTPUT AND WRITE TO FILE
"""

if not os.path.exists(outfolder):
    os.makedirs(outfolder)

# write to file
outfile = "{}/results.txt".format(outfolder)
with open(outfile,'w') as f:
    title = "{:30}{:10}{:10}{}".format('model','mean','std','detail')
    # write lasso
    f.write(title+'\n')
    msg = "{:30}{:<10}{:<10}{}".format('lasso',round(np.mean(res_lasso),3),\
           round(np.std(res_lasso),3),np.round(res_lasso,3))
    f.write(msg+'\n')
    # write ridge
    msg = "{:30}{:<10}{:<10}{}".format('ridge',round(np.mean(res_ridge),3),\
           round(np.std(res_ridge),3),np.round(res_ridge,3))
    f.write(msg+'\n')
    # write Elastic Net
    msg = "{:30}{:<10}{:<10}{}".format('ElasticNet',round(np.mean(res_enet),3),\
           round(np.std(res_enet),3),np.round(res_enet,3))
    f.write(msg+'\n')
    print("results saved to:",outfile)
