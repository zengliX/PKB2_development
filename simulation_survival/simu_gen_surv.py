"""
--------------------------------------------
GENERATE SIMULATION DATASET FOR SURVIVAL
---------------------------------------------
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.getcwd()+'/../simulation_regression/')
from simu_gen_reg import *
"""
--------------------------------------------
FUNTIONS FOR DATA GENERATION
---------------------------------------------
"""

def subsamp(y,col,fold=3):
    grouped = y.groupby(col)
    out = [list() for i in range(fold)]
    for i in grouped.groups.keys():
        ind = grouped.get_group(i)
        n = ind.shape[0]
        r = list(range(0,n+1,n//fold))
        r[-1] = n+1
        # permute index
        perm_index = np.random.permutation(ind.index)
        for j in range(fold):
            out[j] += list(perm_index[r[j]:r[j+1]])
    return out

"""
1.generate survival data from outputs from Fgen_ModelX
clin_data: clinical data generated from Fgen_ModelX
gene_data: gene expression data generated from Fgen_ModelX
pathways: pd.Series of pathways
F: target function value generated from Fgen_ModelX

kappa: scale in baseline weibull hazard
rho: shape in baseline hazard
"""
def Surv_simu_gen(outfolder, clinical, gene, pathways, F,seed=1, censor = 0.2):
    np.random.seed(seed)
    # make directory
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    # save files
    pathways.to_csv(outfolder+'/pathways.txt',index=True)
    clinical.to_csv(outfolder+'/clinical.txt',index_label='sample')
    gene.to_csv(outfolder+'/expression.txt',index_label='sample')
    # generate survival time t
    U = np.random.uniform(size=[len(F),1])
    rho = np.squeeze((np.max(F)-np.min(F))/np.log(100))
    t = (-np.log(U))**(1/rho)/(np.exp(F/rho))
    t *= (20/np.median(t))
    plt.scatter(F.values[:,0],t.values[:,0])
    # generate censor delta
    delta = np.random.binomial(1,0.2,size = [len(F),1])
    y = pd.DataFrame(np.concatenate([t,delta],axis=1),columns = ["survival","censor"])
    y.censor = y.censor.astype('int')
    y.index = F.index
    for i in range(len(F)):
        if y.iloc[i,1] == 1:
            y.iloc[i,0] = np.random.uniform(0,y.iloc[i,0])
    y.to_csv(outfolder+'/response.txt',index_label='sample')
    return y

"""
2. generate files for labels of test data; each file contains 1/3 of all samples
Nfiles: number of files generated
"""
def test_label_gen(outfolder, y, Nfiles= 10, seed = 1):
    np.random.seed(seed)
    for i in range(Nfiles):
        temp = subsamp(y,'censor')
        sele  = temp[0]
        f = "{}/test_label{}.txt".format(outfolder,i)
        with open(f,'w') as thefile:
            for x in sele:
                thefile.write("{}\n".format(x))


"""
--------------------------------------------
GENERATE DATA FOR SURVIVAL
---------------------------------------------
"""
if __file__ == "__main__":
    # simu1
    Nsamp = 300
    Npathway = 20
    Ncateg = 2
    Ncont = 3
    Ng_per_pway = 5
    outfolder = "Surv1_M20"

    clinical, gene, pathways, F = Fgen_Model1(Nsamp,Npathway,Ncateg,Ncont,Ng_per_pway)
    y = Surv_simu_gen(outfolder, clinical, gene, pathways, F)
    test_label_gen(outfolder,y)

    # simu2
    Nsamp = 300
    Npathway = 50
    Ncateg = 2
    Ncont = 3
    Ng_per_pway = 5
    outfolder = "Surv1_M50"

    clinical, gene, pathways, F = Fgen_Model1(Nsamp,Npathway,Ncateg,Ncont,Ng_per_pway)
    y = Surv_simu_gen(outfolder, clinical, gene, pathways, F)
    test_label_gen(outfolder,y)

    # simu3
    Nsamp = 300
    Npathway = 20
    Ncateg = 2
    Ncont = 3
    Ng_per_pway = 5
    outfolder = "Surv2_M20"
    clinical, gene, pathways, F = Fgen_Model2(Nsamp,Npathway,Ncateg,Ncont,Ng_per_pway)
    y = Surv_simu_gen(outfolder, clinical, gene, pathways, F)
    test_label_gen(outfolder,y)

    # simu4
    Nsamp = 300
    Npathway = 50
    Ncateg = 2
    Ncont = 3
    Ng_per_pway = 5
    outfolder = "Surv2_M50"
    clinical, gene, pathways, F = Fgen_Model2(Nsamp,Npathway,Ncateg,Ncont,Ng_per_pway)
    y = Surv_simu_gen(outfolder, clinical, gene, pathways, F)
    test_label_gen(outfolder,y)

    # simu5
    Nsamp = 300
    Npathway = 20
    Ncateg = 2
    Ncont = 3
    Ng_per_pway = 5
    outfolder = "Surv3_M20"
    clinical, gene, pathways, F = Fgen_Model3(Nsamp,Npathway,Ncateg,Ncont,Ng_per_pway)
    y = Surv_simu_gen(outfolder, clinical, gene, pathways, F)
    test_label_gen(outfolder,y)

    # simu6
    Nsamp = 300
    Npathway = 50
    Ncateg = 2
    Ncont = 3
    Ng_per_pway = 5
    outfolder = "Surv3_M50"
    clinical, gene, pathways, F = Fgen_Model3(Nsamp,Npathway,Ncateg,Ncont,Ng_per_pway)
    y = Surv_simu_gen(outfolder, clinical, gene, pathways, F)
    test_label_gen(outfolder,y)
