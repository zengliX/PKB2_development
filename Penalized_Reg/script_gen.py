import os
import argparse
parser = argparse.ArgumentParser(description="Use this command in ~/Penalized_Reg folder, where ~ is the root folder for project")
parser.add_argument("data", help="simulation/real")
parser.add_argument("-c", help="use clinical data only")
parser.add_argument("folders", nargs = '*', help="foldernames")
args = parser.parse_args()

"""
GENEARTE SCRIPT
"""
if args.data == 'simulation':
    folders = [x for x in os.listdir('../simulation_regression') if 'Reg' in x]
    file = 'SimuReg_PenReg.txt'
    with open(file,'w') as f:
        for fd in folders:
            if not os.path.exists(fd):
                os.makedirs(fd)
            start = "source ~/.bashrc; cd /gpfs/ysm/project/lz276/Project3/Penalized_Reg;"
            infolder = '../simulation_regression/{}'.format(fd)
            report = '{}/report.txt'.format(fd)
            f.write("{} python -u PenReg.py {} {} > {} 2>&1\n".format(start, infolder, fd, report))
elif args.data == 'real':
    folders = args.folders
    if args.c is None:
        file = 'realReg_PenReg.txt'
    else:
        file = 'realReg_c_PenReg.txt'
    with open(file,'w') as f:
        for fd in folders:
            infolder = '../data_regression/{}'.format(fd)
            outfolder = fd if args.c is None else "{}_c".format(fd)
            try:
                assert(os.path.exists(infolder))
            except:
                print("{} does not exist".format(infolder))
                continue
            start = "source ~/.bashrc; cd /gpfs/ysm/project/lz276/Project3/Penalized_Reg;"
            report = '{}/report.txt'.format(outfolder)
            if args.c is None:
                f.write("{} python -u PenReg.py {} {} > {} 2>&1\n".format(start, infolder, outfolder, report))
            else:
                f.write("{} python -u PenReg.py {} {} -c True > {} 2>&1\n".format(start, infolder, outfolder, report))
            if not os.path.exists(outfolder):
                os.makedirs(outfolder)
