"""
PLOT WEIGHTS FOR SIMULAITON AND REAL DATA
GENERATE SORTED PATHWAY WEIGHTS FILE FOR REAL DATA
run in ~/analysis/ folder
"""

"""
COMMANDLINE ARGUMENTS
"""
import argparse
parser = argparse.ArgumentParser(description="Use this command in ~/analysis folder, where ~ is the root folder for project")
parser.add_argument("data", help="simulation/real")
parser.add_argument("model", help="classification/regression/survival")
parser.add_argument("folders", nargs = '*', help="foldernames")
args = parser.parse_args()

assert(args.data in ['simulation','real'])
assert(args.model in ['regression','survival'])

"""
IMPORT
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



#data='real'
#model = 'regression'
#folders= ['HDAC']

data = args.data
model = args.model
folders = args.folders
folders.sort()

for fd in folders:
    full_fd = getPath(data,model,fd)
    print("working on folder: {}".format(full_fd))
    if not os.path.exists(full_fd):
        print(full_fd+' does not exist')
        continue
    # get optimal parameter
    opt_par = opt_pars(fd)
    # get weights
    clin_weigh = []
    path_weigh = []
    clin_names = None
    path_names = None
    for i in range(10):
        # load pickled file (return None if file empty)
        pckl = load_pickle(full_fd,opt_par,i)
        if pckl is None:
            print("{}/{}/test_label{}/results.pckl has size 0".format(full_fd,opt_par,i))
            continue
        t = len(pckl.model.test_loss)-1
        #pckl.weights_timeT = MethodType(weights_timeT,pckl)
        w_path, w_clin = pckl.weights_timeT(t)
        if model == 'regression':
            w_clin = w_clin[:-1]
        clin_weigh.append(w_clin)
        path_weigh.append(w_path)
        if clin_names is None:
            clin_names = pckl.inputs.clin_names
        if path_names is None:
            path_names = pckl.inputs.group_names
    #loss_df[fd] = loss
    clin_df = pd.DataFrame(clin_weigh,columns = clin_names)
    path_df = pd.DataFrame(path_weigh,columns = path_names)

    # make output directory
    if not os.path.exists('./PKB2/{}'.format(fd)):
        print("making directory PKB2/{}".format(fd))
        os.makedirs('./PKB2/{}'.format(fd))

    if data == 'real':
        """
        REAL DATA WEIGHTS LIST
        """
        # clinical list (top 10)
        ave_clin = clin_df.apply(np.mean,axis=0).to_frame()
        ave_clin['abs'] = np.abs(ave_clin.iloc[:,0])
        top_clin = ave_clin.sort_values(by='abs',ascending=False).iloc[:,0][:10]
        top_clin.to_csv('./PKB2/{}/top_clinical.txt'.format(fd),index_label='variable')
        clin_df = clin_df[top_clin.index]
        # pathway list (top 20)
        ave_path = path_df.apply(np.mean,axis=0)
        top_path = ave_path.sort_values(ascending=False)[:20]
        top_path.to_csv('./PKB2/{}/top_pathway.txt'.format(fd),index_label='pathway')
        path_df = path_df[top_path.index]

    # clinical plot
    f1 = plt.figure()
    clin_df.boxplot(grid = False)
    plt.axhline(y=0, alpha=0.3, linestyle='-')
    #plt.xticks(rotation=90)
    plt.xticks([],[])
    plt.title(fd)
    plt.xlabel('clinical variable')
    plt.ylabel('weights')
    plt.savefig("PKB2/{}/weights_clinical.pdf".format(fd))
    # pathway plot
    f2 = plt.figure()
    path_df.boxplot(grid = False)
    plt.axhline(y=0, alpha=0.3, linestyle='-')
    #plt.xticks(rotation=90)
    plt.xticks([],[])
    plt.title(fd)
    plt.xlabel('pathways')
    plt.ylabel('weights')
    plt.savefig("PKB2/{}/weights_pathway.pdf".format(fd))
