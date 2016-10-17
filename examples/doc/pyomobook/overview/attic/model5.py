import pyomo.environ
from pyomo.core import *


class Simple(AbstractModel):

    def __init__(self, *args, **kwargs):
        #
        # Initialize base class
        #
        AbstractModel.__init__(self, *args, **kwargs)
        #
        # Declare model components
        #
        self.N = Set()
        self.M = Set()
        self.c = Param(self.N)
        self.a = Param(self.N, self.M)
        self.b = Param(self.M)
        self.x = Var(self.N, within=NonNegativeReals)
        self.obj = Objective()
        self.con = Constraint(self.M)

    def initialize(self, N=[], M=[], c={}, a={}, b={}):
        self.N.initialize(N)
        self.M.initialize(M)
        self.c.initialize(c)
        self.a.initialize(a)
        self.b.initialize(b)

def obj_rule(model):
    return sum(model.c[i]*model.x[i] for i in model.N)
def con_rule(model, m):
    return sum(model.a[i,m]*model.x[i] for i in model.N) \
                    >= model.b[m]
