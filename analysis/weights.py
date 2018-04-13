"""
PLOT WEIGHTS FOR SIMULAITON AND REAL DATA
run in ~/analysis/ folder
"""

from utility import *
import os
from types import MethodType
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append("../PKB2")
import assist


def weight_calc(mat):
    weights = []
    for j in range(mat.shape[1]):
        weights.append( np.sqrt((mat[:,j]**2).sum()) )
    return weights

def weights_timeT(self,t):
        self.coef_mat.fill(0)
        self.coef_clinical = np.zeros(self.inputs.Npred_clin+1)
        # calculate coefficient matrix at step t
        for i in range(1,t+1):
            [m,beta,gamma] = self.model.trace[i]
            self.coef_mat[:,m] += beta*self.inputs.nu
            self.coef_clinical += gamma*self.inputs.nu
        # calculate pathway weights
        weights = weight_calc(self.coef_mat)
        return [weights,self.coef_clinical]

"""
SIMULATION REGRESSION WEIGHTS AND LOSS
"""
folders = [x for x in os.listdir('../PKB2/simu_results') if 'Reg' in x]
folders.sort()
loss_df = pd.DataFrame(np.zeros([10,0]))

for fd in folders:
    print("working on folder: {}".format(fd))
    # get optimal parameter
    opt_par = opt_pars(fd)
    # get precision
    loss = []
    clin_weigh = []
    path_weigh = []
    for i in range(10):
        pckl = load_pickle('../PKB2/simu_results/'+fd,opt_par,i)
        pckl.model.trace[0][2]
        loss.append(pckl.model.test_loss[-1])
        t = len(pckl.model.test_loss)-1
        pckl.weights_timeT = MethodType(weights_timeT,pckl)
        w_path, w_clin = pckl.weights_timeT(t)
        w_clin = w_clin[:-1]
        clin_weigh.append(w_clin)
        path_weigh.append(w_path)
    loss_df[fd] = loss
    clin_df = pd.DataFrame(clin_weigh)
    path_df = pd.DataFrame(path_weigh)
    # clinical plot
    f1 = plt.figure()
    clin_df.boxplot(grid = False)
    plt.axhline(y=0, alpha=0.3, linestyle='-')
    plt.title(fd)
    plt.xlabel('clinical variable')
    plt.ylabel('weights')
    plt.savefig("PKB2/{}/weights_clinical.pdf".format(fd))
    # pathway plot
    f2 = plt.figure()
    path_df.boxplot(grid = False)
    plt.axhline(y=0, alpha=0.3, linestyle='-')
    plt.title(fd)
    plt.xlabel('pathways')
    plt.ylabel('weights')
    plt.savefig("PKB2/{}/weights_pathway.pdf".format(fd))
# save loss_df
loss_df.to_csv("PKB2/regression_loss.csv")    


"""
SIMULATION SURVIVAL WEIGHTS AND LOSS
"""


