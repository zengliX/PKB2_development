"""
Base model class
author: li zeng
"""

import numpy as np

"""
Model class is used to 
- initialize parameters
- keep track of each iteration in boosting
- at the end of program, will be used for visualization
- only the outcome (ytrain, ytest) and predicting function (Ftrain, Ftest) will be stored
"""
class BaseModel:
    """
    model initialization
    coef_mat: beta coefficients for pathways, shape (Ntrain, Ngroup)
    trace: increment parameters in each boosting iteration
    train_loss: training loss of each iteration
    test_loss: testing loss of each iteration
    hasTest: bool, indicating existence of test data
    """
    def __init__(self,inputs,ytrain,ytest):
        self.ytrain = ytrain
        self.ytest = ytest
        self.Ntrain = inputs.Ntrain
        self.Ntest = inputs.Ntest
        self.hasTest = not inputs.Ntest is None
        self.F0 = None
        
        # tracking of performance
        self.coef_mat =  np.zeros([inputs.Ntrain,inputs.Ngroup])
        self.trace = []  # keep track of each iteration
        self.train_loss = [] # loss function at each iteration
        self.test_loss = []  
        return
    
    """
    initialize F_train, ytrain, F_test, ytest
    """
    def init_F(self):
        pass


