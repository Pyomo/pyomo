import pyutilib.th as unittest
from pyomo.environ import *



class MultistartTests(unittest.TestCase):

    def test_as_good_with interation(self):
        with SolverFactory('multistart',)




def build_model():
    model = ConcreteModel()
    model.x1 = Var(initialize = 80,bounds=(0,1000))


    def obj_rule(amodel):
        return model.x1 * sin(model.x1)
    model.obj = Objective(rule=obj_rule,sense=maximize)
    return model
m = main()
