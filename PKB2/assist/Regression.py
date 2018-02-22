from assist.Model import BaseModel
from assist.util import undefined

class PKB_Regression(BaseModel):
    def __init__(self,inputs,ytrain,ytest):
        super().__init__(inputs,ytrain,ytest)
        # for regression only
        self.problem = 'regression'

    """
    initialize regression model
    """
    def init_F(self):
        undefined()

    """
    regression loss function, MSE
    """
    def loss_fun(self,y,f):
        return np.mean( (y-f)**2 )
