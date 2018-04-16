"""
UTILITY FUNCTIONS USED IN ANALYSIS OF PKB2 RESULTS
"""
import pandas as pd
import os

"""
GET OPTIMAL PARAMETERS FOR ONE DATASET
NEED RESULTS IN a results.txt FILE
folder: folder name under ~/analysis/PKB2
"""
def opt_pars(folder):
    file = "./PKB2/{}/results.txt".format(folder)
    with open(file,'r') as f:
        f.readline()
        line = f.readline()
        return line.split(' ')[0]

"""
LOAD THE RESULTS FOR ONE SET OF PARAMETERS FOR A DATA SET
folder: path to the results folder, relatvie to ~/analysis; eg. '../PKB2/simu_results/Reg1_M20'
pars: label for parameters; eg. poly3-L1-0.01-0.2-clinical
test: test number, 0-9
"""
#folder = '../PKB2/simu_results/Reg1_M20'
#pars = opt_pars('Reg1_M20')

def load_pickle(folder,pars,test):
    # folder that has result for pars
    file = "{}/{}/test_label{}/results.pckl".format(folder,pars,test)
    if os.stat(file).st_size > 0:
        res = pd.read_pickle(file)
    else:
        res = None
    return res

"""
get path of output folder (relative to ~/analysis folder)
depending on simu/real, type of analysis, and name of folder
data: simulation/real
model: regression/survival
name: name of the folder
"""
def getPath(data,model,name):
    if data == 'simulation':
        return '../PKB2/simu_results/{}'.format(name)
    elif data == 'real':
        if model == 'regression':
            return '../PKB2/reg_results/{}'.format(name)
        elif model == 'survival':
            return '../PKB2/surv_results/{}'.format(name)
    else:
        return None
