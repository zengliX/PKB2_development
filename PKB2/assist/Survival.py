from assist.Model import BaseModel
from assist.util import undefined

class PKB_Survival(BaseModel):
    def __init__(self,inputs,ytrain,ytest):
        super().__init__(inputs,ytrain,ytest)
        # for survival only
        self.problem = 'survival'

    """
    initialize survival model
    """
    def init_F(self):
        undefined()

    """
    calculate first order derivative
    """
    def calcu_h(self):
        undefined()

    """
    calculate second order derivative
    """
    def calcu_q(self):
        undefined()

    """
    survival loss function, negative log-likelihood
    """
    def loss_fun(self,y,f):
        undefined()

    """
    calculate eta，W, W^(1/2) from h and q
    """
    def calcu_eta(self,h,q):
        undefined()

    def calcu_w(self,q):
        undefined()

    def calcu_w_half(self,q):
        undefined()
