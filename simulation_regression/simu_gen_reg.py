"""
--------------------------------------------
GENERATE SIMULATION DATASET FOR REGRESSION
---------------------------------------------
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

"""
--------------------------------------------
UTIL FUNCTIONS
---------------------------------------------
"""

"""
generate clinical predictors
Nsamp: number of samples
Ncateg: number of binary variables
Ncont: number of continuous variables
"""

def clin_gen(Nsamp, Ncateg, Ncont):
    assert Ncateg>0 and Ncont>0
    categ = pd.DataFrame(np.random.binomial(1,0.5,size = [Nsamp,Ncateg]))
    cont = pd.DataFrame(np.random.normal(0,1,size = [Nsamp,Ncont]))
    out = pd.concat([categ,cont],axis=1)
    out.columns = ["var"+str(i) for i in range(Ncateg+Ncont)]
    out.index = ["sample"+str(i) for i in range(Nsamp)]
    return out

"""
generate gene expression predictors
Nsamp: number of samples
Ngene: number of genes
"""
def gene_gen(Nsamp, Ngene):
    assert Ngene>0
    gene = pd.DataFrame(np.random.normal(0,1,size = [Nsamp,Ngene]))
    gene.columns = ["gene"+str(i) for i in range(Ngene)]
    gene.index = ["sample"+str(i) for i in range(Nsamp)]
    return gene

"""
get pathways gene expressions from the whole gene expression dataframe
gene: the whole gene expression
pw: name of pathway
pathways: pd.Series (a list) of pathways
"""
def get_group_expr(gene,pw,pathways):
    g = pathways[pw].split(' ')
    return gene[g]

"""
--------------------------------------------
FUNTIONS FOR GENERATING F values
---------------------------------------------
"""

"""
Nsamp: number of samples
Npathway: number of pathways
Ncateg: number of categorical clinical variables
Ncont: number of continuous clinical variables
Ng_per_pway: number of genes per pathway
return: clin_data, gene_data, pathways, F
"""


"""
generate clinical data table, gene data table, and pathway pd.Series
"""
def matrix_gen(Nsamp, Npathway, Ncateg, Ncont, Ng_per_pway,seed=1):
    np.random.seed(seed)
    # generate clinical and gene data
    clinical = clin_gen(Nsamp, Ncateg, Ncont).round(3)
    gene = gene_gen(Nsamp,Npathway*Ng_per_pway).round(3)
    # generate pathway list
    temp = dict()
    for i in range(Npathway):
        a = i*Ng_per_pway
        temp[i] = ' '.join( ['gene'+str(j) for j in range(a,a+Ng_per_pway)] )
    pathways = pd.Series(temp)
    pathways.index = ['group'+str(x) for x in pathways.index]
    return clinical, gene, pathways


"""
visualize the contribution from different parts
"""
def simu_summary(parts,F):
    print(np.var(parts))
    for i in range(parts.shape[1]):
        plt.scatter(parts[i].values,F.values)
        plt.title("F vs part"+str(i))
        plt.show()


def Fgen_Model1(Nsamp, Npathway, Ncateg, Ncont, Ng_per_pway,seed=1):
    clinical, gene, pathways = matrix_gen(Nsamp, Npathway, Ncateg, Ncont, Ng_per_pway,seed)
    # generate F
        # clinical part
    coefs = np.array([3,-4,3])
    part0 = clinical.iloc[:,:3].dot(coefs)
        # pathway1
    temp = get_group_expr(gene,"group0",pathways)
    part1 = temp.iloc[:,:2].dot([2,3])
        # pathway2
    temp = get_group_expr(gene,"group1",pathways)
    part2 = 3*np.exp(temp.iloc[:,:2].dot([0.5,0.5]))
        # pathway3
    temp = get_group_expr(gene,"group2",pathways)
    part3 = 4*temp.iloc[:,0]*temp.iloc[:,1]
    parts = pd.concat([part0,part1,part2,part3],axis=1)
    F = parts.apply(sum,axis=1).to_frame(name='response')
    F = F - F.median()
    simu_summary(parts,F)
    return clinical, gene, pathways, F

def Fgen_Model2(Nsamp, Npathway, Ncateg, Ncont, Ng_per_pway,seed=1):
    clinical, gene, pathways = matrix_gen(Nsamp, Npathway, Ncateg, Ncont, Ng_per_pway,seed)
    # generate F
        # clinical part
    coefs = np.array([1,-3,3,-1])
    part0 = clinical.iloc[:,:4].dot(coefs)
        # pathway1
    temp = get_group_expr(gene,"group0",pathways)
    part1 = 6*np.sin(temp.iloc[:,:2].dot([.5,.5]))
        # pathway2
    temp = get_group_expr(gene,"group1",pathways)
    part2 = 2*np.log(np.abs(temp.iloc[:,0]**3 - temp.iloc[:,1]**3))
        # pathway3
    temp = get_group_expr(gene,"group2",pathways)
    part3 = 2*(temp.iloc[:,0]**2 - temp.iloc[:,1]**2)
    parts = pd.concat([part0,part1,part2,part3],axis=1)
    F = parts.apply(sum,axis=1).to_frame(name='response')
    F = F - F.median()
    simu_summary(parts,F)
    return clinical, gene, pathways, F

def Fgen_Model3(Nsamp, Npathway, Ncateg, Ncont, Ng_per_pway,seed=1):
    clinical, gene, pathways = matrix_gen(Nsamp, Npathway, Ncateg, Ncont, Ng_per_pway,seed)
    # generate F
        # clinical part
    coefs = np.array([2,0,2])
    parts = np.zeros([Nsamp,9])
    parts[:,0] = clinical.iloc[:,:3].dot(coefs)
        # gene part
    for i in range(8):
        temp = get_group_expr(gene,"group{}".format(i),pathways)
        parts[:,i+1] = 2*np.sqrt((temp**2).apply(sum,axis=1))
    parts = pd.DataFrame(parts,index = clinical.index)
    F = parts.apply(sum,axis=1).to_frame(name='response')
    F = F - F.median()
    simu_summary(parts,F)
    return clinical, gene, pathways, F

"""
--------------------------------------------
FUNTIONS FOR DATA GENERATION
---------------------------------------------
"""

"""
1.generate regression data from outputs from Fgen_ModelX
clin_data: clinical data generated from Fgen_ModelX
gene_data: gene expression data generated from Fgen_ModelX
pathways: pd.Series of pathways
F: target function value generated from Fgen_ModelX
"""
def Reg_simu_gen(outfolder, clinical, gene, pathways, F,seed=1):
    np.random.seed(seed)
    # make directory
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    # save files
    pathways.to_csv(outfolder+'/pathways.txt',index=True)
    clinical.to_csv(outfolder+'/clinical.txt',index_label='sample')
    gene.to_csv(outfolder+'/expression.txt',index_label='sample')
    # add regression noise (account for 10% of variance) to F
    v = np.var(F)
    y = F + np.random.normal(scale = np.sqrt(v/5),size=F.shape)
    y.to_csv(outfolder+'/response.txt',index_label='sample')


"""
2. generate files for labels of test data; each file contains 1/3 of all samples
Nfiles: number of files generated
"""
def test_label_gen(outfolder, clinical, Nfiles= 10, seed = 1):
    np.random.seed(seed)
    labels = clinical.index
    Ntest = len(labels)//3
    for i in range(Nfiles):
        sele = np.random.choice(labels,Ntest,replace=False)
        f = "{}/test_label{}.txt".format(outfolder,i)
        with open(f,'w') as thefile:
            for x in sele:
                thefile.write("{}\n".format(x))

"""
--------------------------------------------
GENERATE DATA FOR REGRESSION
---------------------------------------------
"""
if __file__ == "__main__":
    # simu1
    Nsamp = 300
    Npathway = 20
    Ncateg = 2
    Ncont = 3
    Ng_per_pway = 5
    outfolder = "Reg1_M20"

    clinical, gene, pathways, F = Fgen_Model1(Nsamp,Npathway,Ncateg,Ncont,Ng_per_pway)
    Reg_simu_gen(outfolder, clinical, gene, pathways, F)
    test_label_gen(outfolder,clinical)

    # simu2
    Nsamp = 300
    Npathway = 50
    Ncateg = 2
    Ncont = 3
    Ng_per_pway = 5
    outfolder = "Reg1_M50"

    clinical, gene, pathways, F = Fgen_Model1(Nsamp,Npathway,Ncateg,Ncont,Ng_per_pway)
    Reg_simu_gen(outfolder, clinical, gene, pathways, F)
    test_label_gen(outfolder,clinical)

    # simu3
    Nsamp = 300
    Npathway = 20
    Ncateg = 2
    Ncont = 3
    Ng_per_pway = 5
    outfolder = "Reg2_M20"
    clinical, gene, pathways, F = Fgen_Model2(Nsamp,Npathway,Ncateg,Ncont,Ng_per_pway, seed= 1)
    Reg_simu_gen(outfolder, clinical, gene, pathways, F)
    test_label_gen(outfolder,clinical)

    # simu4
    Nsamp = 300
    Npathway = 50
    Ncateg = 2
    Ncont = 3
    Ng_per_pway = 5
    outfolder = "Reg2_M50"
    clinical, gene, pathways, F = Fgen_Model2(Nsamp,Npathway,Ncateg,Ncont,Ng_per_pway,seed= 1)
    Reg_simu_gen(outfolder, clinical, gene, pathways, F)
    test_label_gen(outfolder,clinical)

    # simu5
    Nsamp = 300
    Npathway = 20
    Ncateg = 2
    Ncont = 3
    Ng_per_pway = 5
    outfolder = "Reg3_M20"
    clinical, gene, pathways, F = Fgen_Model3(Nsamp,Npathway,Ncateg,Ncont,Ng_per_pway)
    Reg_simu_gen(outfolder, clinical, gene, pathways, F)
    test_label_gen(outfolder,clinical)

    # simu6
    Nsamp = 300
    Npathway = 50
    Ncateg = 2
    Ncont = 3
    Ng_per_pway = 5
    outfolder = "Reg3_M50"
    clinical, gene, pathways, F = Fgen_Model3(Nsamp,Npathway,Ncateg,Ncont,Ng_per_pway)
    Reg_simu_gen(outfolder, clinical, gene, pathways, F)
    test_label_gen(outfolder,clinical)
