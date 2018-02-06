from assist.Model import BaseModel

class PKB_Survival(BaseModel):
    def __init__(self,inputs,ytrain,ytest):
        super().__init__(inputs,ytrain,ytest)
        # for survival only
        self.problem = 'survival'

    """
    initialize survival model
    """
    def init_F(self):
        pass

    """
    calculate first order derivative
    """
    def calcu_h(self):
        pass

    """
    calculate second order derivative
    """
    def calcu_q(self):
        pass

    """
    survival loss function, negative log-likelihood
    """
    def loss_fun(self,y,f):
        pass
