import os
import argparse
parser = argparse.ArgumentParser(description="Use this command in ~/RandomForest folder, where ~ is the root folder for project")
parser.add_argument("data", help="simulation/real")
parser.add_argument("folders", nargs = '*', help="foldernames")
args = parser.parse_args()

"""
GENEARTE SIMU REGRESSION SCRIPT
"""
if args.data == 'simulation':
    folders = [x for x in os.listdir('../simulation_regression') if 'Reg' in x]
    file = 'SimuReg_RF.txt'
    with open(file,'w') as f:
        for fd in folders:
            if not os.path.exists(fd):
                os.makedirs(fd)
            start = "source ~/.bashrc; cd /gpfs/ysm/project/lz276/Project3/RandomForest;"
            infolder = '../simulation_regression/{}'.format(fd)
            report = '{}/report.txt'.format(fd)
            f.write("{} python -u RandomForest.py {} {} > {} 2>&1\n".format(start, infolder, fd, report))
elif args.data == 'real':
    folders = args.folders
    file = 'realReg_RF.txt'
    with open(file,'w') as f:
        for fd in folders:
            infolder = '../data_regression/{}'.format(fd)
            try:
                assert(os.path.exists(infolder))
            except:
                print("{} does not exist".format(infolder))
                continue
            start = "source ~/.bashrc; cd /gpfs/ysm/project/lz276/Project3/RandomForest;"
            report = '{}/report.txt'.format(fd)
            f.write("{} python -u RandomForest.py {} {} > {} 2>&1\n".format(start, infolder, fd, report))
            if not os.path.exists(fd):
                os.makedirs(fd)
