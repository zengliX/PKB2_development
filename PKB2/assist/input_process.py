"""
source file for input parameter processing
author: li zeng
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

# test existence of data
def have_file(myfile):
    if not os.path.exists(myfile):
            print("file:",myfile,"does not exist")
            sys.exit(-1)
    else:
        print("reading file:",myfile)

class input_obj:
    """
    parse command line arguments
    import data
    """
    "class object for input data"
    # data status
    loaded = False

    # type of problem
    problem = None

    #folder/files
    input_folder = None
    output_folder= None
    group_file = None
    train_predictor_file = None
    train_response_file = None

    # input, output pars
    train_predictors=None
    train_response=None
    pred_sets=None

    # optional
    test_file = None
    test_predictors = None
    test_response = None

    # input summary
    Ngroup = None
    Ntrain = None
    Ntest = None
    Npred = None # number of predictor
    group_names = None

    # model pars
    nu = 0.05
    maxiter = 800
    Lambda = None
    kernel = None
    method = None
    pen = 1

    # initialize with command line arguments
    def __init__(self,args):
        # assign to class objects
        self.problem = args.problem
        self.input_folder = args.input
        self.output_folder = args.output
        self.train_predictor_file = args.predictor
        self.train_response_file = args.response
        self.group_file = args.predictor_set
        self.kernel = args.kernel
        self.method = args.method
        if not args.maxiter is None: self.maxiter= int(args.maxiter)
        if not args.rate is None: self.nu = float(args.rate)
        if not args.Lambda is None: self.Lambda = float(args.Lambda)
        if not args.test is None: self.test_file = args.test
        if not args.pen is None: self.pen = float(args.pen)
        #print(self.__dict__)

    # ██████  ██████   ██████   ██████     ██ ███    ██ ██████  ██    ██ ████████
    # ██   ██ ██   ██ ██    ██ ██          ██ ████   ██ ██   ██ ██    ██    ██
    # ██████  ██████  ██    ██ ██          ██ ██ ██  ██ ██████  ██    ██    ██
    # ██      ██   ██ ██    ██ ██          ██ ██  ██ ██ ██      ██    ██    ██
    # ██      ██   ██  ██████   ██████     ██ ██   ████ ██       ██████     ██

    # function to process input
    def proc_input(self):
        """
        load corresponding data
        """
        print('----------- LOAD DATA -------------------')
        # make output folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # training data
        thisfile = self.input_folder +"/"+ self.group_file
        have_file(thisfile)
        self.pred_sets = pd.Series.from_csv(thisfile)

        thisfile = self.input_folder + "/"+ self.train_predictor_file
        have_file(thisfile)
        self.train_predictors = pd.DataFrame.from_csv(thisfile)

        thisfile = self.input_folder + "/"+ self.train_response_file
        have_file(thisfile)
        self.train_response = pd.DataFrame.from_csv(thisfile)

        # data summary
        self.Ntrain = self.train_predictors.shape[0]
        self.Ngroup = self.pred_sets.shape[0]
        self.Npred = self.train_predictors.shape[1]
        self.group_names = self.pred_sets.index

        # change loaded indicator
        self.loaded = True
        return

    def data_preprocessing(self,center = False,norm=False):
        """
        preprocess data:
        remove low variance, normalize each column, and drop not used groups
        """
        print()
        print('------------PROCESS DATA ----------------')
        if not self.loaded:
            print("No data loaded. Can not preprocess.")
            return

        # center data
        if center:
            print('Centering data.')
            scale(self.train_predictors,copy=False,with_std=False)

        # normalize data
        if norm:
            print("Normalizing data.")
            scale(self.train_predictors,copy=False,with_mean=False)

        # check groups
        print("Checking groups.")
        to_drop =[]
        for i in range(len(self.pred_sets)):
            genes = self.pred_sets.values[i].split(" ")
            shared = np.intersect1d(self.train_predictors.columns.values,genes)
            if len(shared)==0:
                print("Drop group:",self.pred_sets.index[i])
                to_drop.append(i)
            else:
                self.pred_sets.values[i] = ' '.join(shared)
        if len(to_drop)>0:
            self.pred_sets = self.pred_sets.drop(self.pred_sets.index[to_drop])

        # calculate summary
        self.Ngroup = len(self.pred_sets)
        return


    def data_split(self):
        """
        split the data into test and train
        """
        print()
        print('--------------SPLIT DATA ----------------')
        print("Using test label: ",self.test_file)
        if self.test_file is None: return
        # load test file
        thisfile = self.input_folder+'/'+self.test_file
        f  = open(thisfile,'r')
        test_ind = [x.strip() for x in f]
        f.close()
        # split data
        self.test_predictors = self.train_predictors.loc[test_ind]
        self.test_response = self.train_response.loc[test_ind]
        train_ind = np.setdiff1d(self.train_predictors.index.values,np.array(test_ind))
        self.train_predictors = self.train_predictors.loc[train_ind]
        self.train_response = self.train_response.loc[train_ind]
        # update summary
        self.Ntest = len(self.test_response)
        self.Ntrain = len(self.train_response)

    # ██████  ██████  ██ ███    ██ ████████
    # ██   ██ ██   ██ ██ ████   ██    ██
    # ██████  ██████  ██ ██ ██  ██    ██
    # ██      ██   ██ ██ ██  ██ ██    ██
    # ██      ██   ██ ██ ██   ████    ██

    def input_summary(self):
        print()
        print('------------- SUMMARY ------------------')
        print("Analysis type:",self.problem)
        print("input folder:", self.input_folder)
        print("output file folder:",self.output_folder)
        print("number of training samples:",self.Ntrain)
        print("number of test samples:",self.Ntest)
        print("number of groups:", self.Ngroup)
        print("number of predictors:", self.Npred)
        return

    def model_param(self):
        print()
        print('------------ PARAMS --------------------')
        print("learning rate:",self.nu)
        print("Lambda:",self.Lambda)
        print("maximum iteration:", self.maxiter)
        print("kernel function: ",self.kernel)
        print("method: ",self.method)
        return
