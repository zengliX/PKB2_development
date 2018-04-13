"""
COLLECT PKB RESUTLS AND CREATE SORTED TABLE
run in ~/analysis folder
"""

# arguments
import argparse
parser = argparse.ArgumentParser(description="Use this command in ~/analysis folder, where ~ is the root folder for project")
parser.add_argument("data", help="simulation/real")
parser.add_argument("model", help="classification/regression/survival")
parser.add_argument("folders", nargs = '*', help="foldernames")
args = parser.parse_args()

# import
import os
import sys
import pickle
import pandas as pd
import numpy as np

sys.path.append("../PKB2")
import assist

"""
FUNCTION FOR EVALUATE THE RESULTS FOR ONE DATASET
infdlder: path to folder that contains results
outfolder: path to output folder
"""

def PKBanalyze(infolder,outfolder):
    print("Analyzing folder: {}".format(infolder))
    fds = [x for x in os.listdir(infolder) if '-' in x]
    res = []
    for f in fds:
        # results for one parameter set
        res_f = []
        cur_fd = "{}/{}".format(infolder,f)
        # pickle file in each test folder
        test_f = [ cur_fd+'/'+x+'/'+"results.pckl" for x in os.listdir(cur_fd) if 'test' in x]
        for one_test in test_f:
            try:
                saved = pd.read_pickle(one_test)
                if saved.model.problem == "regression":
                    res_f.append(saved.model.test_loss[-1])
                elif saved.model.problem == "survival":
                    res_f.append(saved.model.test_Cind[-1])
            except:
                continue
        res.append([np.mean(res_f), np.std(res_f), res_f, f]) # mean, std, list, parameter
    # sort results and write to file
    sort_results(res,outfolder,ascending = (saved.model.problem == 'regression') )

"""
sort PKB results from best to worst, and save to file
res: the res variable from PKBanalyze
outfolder: output folder
"""
def sort_results(res,outfolder,ascending = True):
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    move_nan(res)
    res.sort()
    if not ascending:
        res.reverse()
        move_nan(res)
    outfile = "{}/results.txt".format(outfolder)
    with open(outfile,'w') as file:
        title = "{:40}{:10}{:10}{}".format('parameters','mean','std','detail')
        file.write(title+'\n')
        for x in res:
            m, sd, vals, pars = x
            msg = "{:40}{:<10}{:<10}{}".format(pars,round(m,3),round(sd,3),np.round(vals,3))
            file.write(msg+'\n')

def move_nan(ls):
    """
    move nan to end of a list
    """
    p1 = p2 = 0
    for p2 in range(len(ls)):
        if not np.isnan(ls[p2][0]) and p1!=p2:
            ls[p1], ls[p2] = ls[p2], ls[p1]
            p1 += 1


# ███    ███  █████  ██ ███    ██
# ████  ████ ██   ██ ██ ████   ██
# ██ ████ ██ ███████ ██ ██ ██  ██
# ██  ██  ██ ██   ██ ██ ██  ██ ██
# ██      ██ ██   ██ ██ ██   ████

if __name__ == "__main__":
    #infolder = "../PKB2/simu_results/Reg1_M20"
    #outfolder = "./PKB2/Reg1_M20"
    fds = args.folders

    for x in fds:
        if args.data == 'simulation':
            infolder = "../PKB2/simu_results/{}".format(x)
        elif args.data == 'real' and args.model == 'regression':
            infolder = "../PKB2/reg_results/{}".format(x)
        elif args.data == 'real' and args.model == 'survival':
            infolder = "../PKB2/surv_results/{}".format(x)
        outfolder = './PKB2/{}'.format(x)
        try:
            assert(os.path.exists(infolder))
        except:
            print("{} does not exist".format(infolder))
            continue
        PKBanalyze(infolder,outfolder)
