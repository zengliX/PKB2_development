"""
utility functions for running models and analysing results
"""
import itertools
import re
import os

"""
write list elements to file, one item per row
filename: file to be written
l: list of strs
"""
def write_list(filename,l):
    with open(filename,'w') as f:
        for x in l:
            f.write(x+'\n')

"""
generate terminal command to run PKB2 for different combinations of parameters
d: dictionary of input parameters

example:
d = {
'model': 'classification',
'input_folder': 'folderIN',
'output_folder': 'folderOUT',
'predictor': 'expression.txt',
'sets': 'pathways.txt',
'response': 'response.txt',
'maxiter': 500,
'kernel': ['poly3', 'rbf'],
'reg': ['L1','L2'],
'rate': [0.01, 0.05],
'pen': [0.2,1,2],
'clinical': 'clinical.txt',
'test': ['test_label0.txt','test_label1.txt']
}

# output folder name: 'kernel-regularization-rate-pen-clinical/test_label'
# run the command inside the PKB2 folder
"""

def simu_command_gen(d):
    out = []
    for k,reg,r,p in itertools.product(d['kernel'],d['reg'],d['rate'],d['pen']):
        label = "{}-{}-{}-{}".format(k,reg,r,p)
        if 'clinical' in d:
            label += '-clinical'
        for Tlab in d['test']:
            Tlab2= re.sub('.txt','',Tlab)
            outfolder = "{}/{}/{}".format(d['output_folder'],label,Tlab2)
            if not os.path.exists(outfolder):
                os.makedirs(outfolder)
            report = "{}/{}".format(outfolder,"report.txt")
            s = "python -u PKB2.py {} {} {} {} {} {} {} {} -maxiter {} -rate {} -pen {}".format(\
                            d['model'], d['input_folder'],outfolder,\
                            d['predictor'], d['sets'], d['response'],\
                            k,reg,d['maxiter'],str(r),str(p))
            s += ' -test {}'.format(Tlab)
            if 'clinical' in d:
                s += ' -clinical {}'.format(d['clinical'])
            s += ' > ' + report + ' 2>&1'
            s= "source ~/.bashrc; cd /gpfs/ysm/project/lz276/Project3/PKB2/; "+s
            out.append(s)
    return out

"""
generate terminal command to run PKB2 for different combinations of parameters
d: dictionary of input parameters

example:
d = {
'model': 'survival',
'input_folder': 'folderIN',
'output_folder': 'folderOUT',
'predictor': 'expression.txt',
'sets': ['KEGG','Biocarta','GO_BP'],
'response': 'response.txt',
'maxiter': 1000,
'kernel': ['poly3', 'rbf'],
'reg': ['L1','L2'],
'rate': [0.005, 0.03],
'pen': [0.2,1,2],
'clinical': 'clinical.txt',
'test': ['test_label0.txt','test_label1.txt']
}

# output folder name: 'kernel-regularization-rate-pen-pathway-clinical/test_label'
# run the command inside the PKB2 folder
"""

def real_command_gen(d):
    out = []
    for k,reg,r,p,group in itertools.product(d['kernel'],d['reg'],d['rate'],d['pen'],d['sets']):
        label = "{}-{}-{}-{}-{}".format(k,reg,r,p,group)
        pathway = "../../PKB2/pathways/{}.txt".format(group)
        if 'clinical' in d:
            label += '-clinical'
        for Tlab in d['test']:
            Tlab2= re.sub('.txt','',Tlab)
            outfolder = "{}/{}/{}".format(d['output_folder'],label,Tlab2)
            if not os.path.exists(outfolder):
                os.makedirs(outfolder)
            report = "{}/{}".format(outfolder,"report.txt")
            s = "python -u PKB2.py {} {} {} {} {} {} {} {} -maxiter {} -rate {} -pen {}".format(\
                            d['model'], d['input_folder'],outfolder,\
                            d['predictor'], pathway, d['response'],\
                            k,reg,d['maxiter'],str(r),str(p))
            s += ' -test {}'.format(Tlab)
            if 'clinical' in d:
                s += ' -clinical {}'.format(d['clinical'])
            s += ' > ' + report + ' 2>&1'
            s= "source ~/.bashrc; cd /gpfs/ysm/project/lz276/Project3/PKB2/; "+s
            out.append(s)
    return out


# ███    ███  █████  ██ ███    ██
# ████  ████ ██   ██ ██ ████   ██
# ██ ████ ██ ███████ ██ ██ ██  ██
# ██  ██  ██ ██   ██ ██ ██  ██ ██
# ██      ██ ██   ██ ██ ██   ████

if __name__ == "__main__":
    # arguments
    import argparse
    parser = argparse.ArgumentParser(description="Use this command in ~/PKB2 folder, where ~ is the root folder for project")
    parser.add_argument("data", help="simulation/real")
    parser.add_argument("model", help="classification/regression/survival")
    parser.add_argument("folders", nargs = '*', help="foldernames")
    args = parser.parse_args()

    folders = args.folders
    """
    regression simulations
    """
    if args.data == 'simulation' and args.model == 'regression':
        for fd in folders:
            infolder = '../simulation_regression/'+fd
            outfolder = './simu_results/'+fd
            try:
                assert(os.path.exists(infolder))
            except:
                print("{} does not exist".format(infolder))
                continue
            d = {
                    'model': 'regression',
                    'input_folder': infolder,
                    'output_folder': outfolder,
                    'predictor': 'expression.txt',
                    'sets': 'pathways.txt',
                    'response': 'response.txt',
                    'maxiter': 800,
                    'kernel': ['poly3', 'rbf'],
                    'reg': ['L1','L2'],
                    'rate': [0.01, 0.05],
                    'pen': [0.04,0.2,1],
                    'clinical': 'clinical.txt',
                    'test': ['test_label{}.txt'.format(x) for x in range(10)]
                    }
            out = simu_command_gen(d)
            write_list('./scripts/{}.txt'.format(fd),out)
    """
    survival simulations
    """
    if args.data == 'simulation' and args.model == 'survival':
        for fd in folders:
            infolder = '../simulation_survival/'+fd
            outfolder = './simu_results/'+fd
            try:
                assert(os.path.exists(infolder))
            except:
                print("{} does not exist".format(infolder))
                continue
            d = {
                'model': 'survival',
                'input_folder': infolder,
                'output_folder': outfolder,
                'predictor': 'expression.txt',
                'sets': 'pathways.txt',
                'response': 'response.txt',
                'maxiter': 800,
                'kernel': ['poly3', 'rbf'],
                'reg': ['L1','L2'],
                'rate': [0.01, 0.05],
                'pen': [0.04,0.2,1],
                'clinical': 'clinical.txt',
                'test': ['test_label{}.txt'.format(x) for x in range(10)]
                }
            out = simu_command_gen(d)
            write_list('./scripts/{}.txt'.format(fd),out)


    """
    real data regression
    """
    if args.data == 'real' and args.model == 'regression':
        for fd in folders:
            infolder = '../data_regression/'+fd
            outfolder = './reg_results/'+fd
            try:
                assert(os.path.exists(infolder))
            except:
                print("{} does not exist".format(infolder))
                continue
            d = {
                'model': 'regression',
                'input_folder': infolder,
                'output_folder': outfolder,
                'predictor': 'expression.txt',
                'sets': ['KEGG','GO_BP','Biocarta'],
                'response': 'response.txt',
                'maxiter': 800,
                'kernel': ['poly3', 'rbf'],
                'reg': ['L1','L2'],
                'rate': [0.005, 0.03],
                'pen': [0.04,0.2,1],
                #'clinical': 'clinical.txt',
                'test': ['test_label{}.txt'.format(x) for x in range(10)]
                }
            out = real_command_gen(d)
            write_list('./scripts/{}.txt'.format(fd),out)

    """
    real data survival
    """
    if args.data == 'real' and args.model == 'survival':
        for fd in folders:
            infolder = '../data_survival/'+fd
            outfolder = './surv_results/'+fd
            try:
                assert(os.path.exists(infolder))
            except:
                print("{} does not exist".format(infolder))
                continue
            d = {
                'model': 'survival',
                'input_folder': infolder,
                'output_folder': outfolder,
                'predictor': 'expression.txt',
                'sets': ['KEGG','GO_BP','Biocarta'],
                'response': 'response.txt',
                'maxiter': 800,
                'kernel': ['poly3', 'rbf'],
                'reg': ['L1','L2'],
                'rate': [0.005, 0.03],
                'pen': [0.04,0.2,1],
                'clinical': 'clinical.txt',
                'test': ['test_label{}.txt'.format(x) for x in range(10)]
                }
            out = real_command_gen(d)
            write_list('./scripts/{}.txt'.format(fd),out)
