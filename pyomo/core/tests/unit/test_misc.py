#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Unit Tests for pyomo.base.misc
#

import re
import os
from os.path import abspath, dirname
currdir= dirname(abspath(__file__))

import pyutilib.th as unittest
from pyutilib.misc import setup_redirect, reset_redirect

from pyomo.opt import check_available_solvers
import pyomo.scripting.pyomo_command as main
from pyomo.core import *

from six import StringIO

def rule1(model):
    return (1,model.x+model.y[1],2)
def rule2(model,i):
    return (1,model.x+model.y[1]+i,2)


solvers = None
class PyomoModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        global solvers
        import pyomo.environ
        solvers = check_available_solvers('glpk', 'cplex')

    def test_construct(self):
        model = AbstractModel()
        model.a = Set(initialize=[1,2,3])
        model.A = Param(initialize=1)
        model.B = Param(model.a)
        model.x = Var(initialize=1, within=Reals, dense=False)
        model.y = Var(model.a, initialize=1, within=Reals, dense=False)
        model.obj = Objective(rule=lambda model: model.x+model.y[1])
        model.obj2 = Objective(model.a,rule=lambda model, i: i+model.x+model.y[1])
        model.con = Constraint(rule=rule1)
        model.con2 = Constraint(model.a, rule=rule2)
        instance = model.create_instance()
        expr = instance.x + 1

        OUTPUT = open(currdir+"/display.out","w")
        display(instance,ostream=OUTPUT)
        display(instance.obj,ostream=OUTPUT)
        display(instance.x,ostream=OUTPUT)
        display(instance.con,ostream=OUTPUT)
        expr.to_string(ostream=OUTPUT)
        model = AbstractModel()
        instance = model.create_instance()
        display(instance,ostream=OUTPUT)
        OUTPUT.close()
        try:
            display(None)
            self.fail("test_construct - expected TypeError")
        except TypeError:
            pass
        self.assertFileEqualsBaseline(currdir+"/display.out",currdir+"/display.txt")

    def test_construct2(self):
        model = AbstractModel()
        model.a = Set(initialize=[1,2,3])
        model.A = Param(initialize=1)
        model.B = Param(model.a)
        model.x = Var(initialize=1, within=Reals, dense=True)
        model.y = Var(model.a, initialize=1, within=Reals, dense=True)
        model.obj = Objective(rule=lambda model: model.x+model.y[1])
        model.obj2 = Objective(model.a,rule=lambda model, i: i+model.x+model.y[1])
        model.con = Constraint(rule=rule1)
        model.con2 = Constraint(model.a, rule=rule2)
        instance = model.create_instance()
        expr = instance.x + 1

        OUTPUT = open(currdir+"/display2.out","w")
        display(instance,ostream=OUTPUT)
        display(instance.obj,ostream=OUTPUT)
        display(instance.x,ostream=OUTPUT)
        display(instance.con,ostream=OUTPUT)
        expr.to_string(ostream=OUTPUT)
        model = AbstractModel()
        instance = model.create_instance()
        display(instance,ostream=OUTPUT)
        OUTPUT.close()
        try:
            display(None)
            self.fail("test_construct - expected TypeError")
        except TypeError:
            pass
        self.assertFileEqualsBaseline(currdir+"/display2.out",currdir+"/display2.txt")


class PyomoBadModels ( unittest.TestCase ):

    @classmethod
    def setUpClass(cls):
        global solvers
        import pyomo.environ
        solvers = check_available_solvers('glpk', 'cplex')

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

    def test_uninstantiated_model_linear ( self ):
        """Run pyomo with "bad" model file.  Should fail gracefully, with
        a perhaps useful-to-the-user message."""
        if not 'glpk' in solvers:
            self.skipTest("glpk solver is not available")
        return # ignore for now
        base = '%s/test_uninstantiated_model' % currdir
        fout, fbase = (base + '_linear.out', base + '.txt')
        self.pyomo('uninstantiated_model_linear.py', file=fout )
        self.assertFileEqualsBaseline( fout, fbase )

    def test_uninstantiated_model_quadratic ( self ):
        """Run pyomo with "bad" model file.  Should fail gracefully, with
        a perhaps useful-to-the-user message."""
        if not 'cplex' in solvers:
            self.skipTest("The 'cplex' executable is not available")
        return # ignore for now
        base = '%s/test_uninstantiated_model' % currdir
        fout, fbase = (base + '_quadratic.out', base + '.txt')
        self.pyomo('uninstantiated_model_quadratic.py --solver=cplex', file=fout )
        self.assertFileEqualsBaseline( fout, fbase )

if __name__ == "__main__":
    unittest.main()
