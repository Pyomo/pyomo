#
# Unit Tests for pyomo.base.misc
#

import os, re, sys
from os.path import abspath, dirname
currdir= dirname(abspath(__file__))

from six import StringIO

from pyutilib.misc import setup_redirect, reset_redirect
from pyutilib.services import registered_executable
import pyutilib.th as unittest
from coopr.opt import load_solvers
from coopr.pyomo import *
import coopr.pyomo.scripting.pyomo as main
import coopr.environ

solver = load_solvers('glpk', 'cplex')


def rule1(model):
    return (1,model.x+model.y[1],2)
def rule2(model,i):
    return (1,model.x+model.y[1]+i,2)

class PyomoModel(unittest.TestCase):

    def setUp(self):
        self.model = AbstractModel()

    def test_construct(self):
        model = AbstractModel()
        model.a = Set(initialize=[1,2,3])
        model.A = Param(initialize=1)
        model.B = Param(model.a)
        model.x = Var(initialize=1,within=Reals)
        model.y = Var(model.a, initialize=1,within=Reals)
        model.obj = Objective(rule=lambda model: model.x+model.y[1])
        model.obj2 = Objective(model.a,rule=lambda model, i: i+model.x+model.y[1])
        model.con = Constraint(rule=rule1)
        model.con2 = Constraint(model.a, rule=rule2)
        instance = model.create()
        expr = instance.x + 1
        #instance.reset()
        OUTPUT = open(currdir+"/display.out","w")
        display(instance,ostream=OUTPUT)
        display(instance.obj,ostream=OUTPUT)
        display(instance.x,ostream=OUTPUT)
        display(instance.con,ostream=OUTPUT)
        expr.to_string(ostream=OUTPUT)
        model = AbstractModel()
        instance = model.create()
        display(instance,ostream=OUTPUT)
        OUTPUT.close()
        try:
            display(None)
            self.fail("test_construct - expected TypeError")
        except TypeError:
            pass
        self.assertFileEqualsBaseline(currdir+"/display.out",currdir+"/display.txt")


class PyomoBadModels ( unittest.TestCase ):

    def pyomo ( self, cmd, **kwargs):
        args = re.split('[ ]+', cmd )
        out = kwargs.get( 'file', None )
        if out is None:
            out = StringIO()
        setup_redirect( out )
        os.chdir( currdir )
        output = main.run( args )
        reset_redirect()
        if not 'file' in kwargs:
            return OUTPUT.getvalue()
        return output

    @unittest.skipIf(solver['glpk'] is None, "glpk solver is not available")
    def test_uninstantiated_model_linear ( self ):
        """Run pyomo with "bad" model file.  Should fail gracefully, with
        a perhaps useful-to-the-user message."""
        return # ignore for now
        base = '%s/test_uninstantiated_model' % currdir
        fout, fbase = (base + '_linear.out', base + '.txt')
        self.pyomo('uninstantiated_model_linear.py', file=fout )
        self.assertFileEqualsBaseline( fout, fbase )

    @unittest.skipIf(solver['cplex'] is None, "The 'cplex' executable is not available")
    def test_uninstantiated_model_quadratic ( self ):
        """Run pyomo with "bad" model file.  Should fail gracefully, with
        a perhaps useful-to-the-user message."""
        return # ignore for now
        base = '%s/test_uninstantiated_model' % currdir
        fout, fbase = (base + '_quadratic.out', base + '.txt')
        self.pyomo('uninstantiated_model_quadratic.py --solver=cplex', file=fout )
        self.assertFileEqualsBaseline( fout, fbase )

if __name__ == "__main__":
    unittest.main()
