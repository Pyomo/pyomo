#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Test that EXPORT type Suffix components work with the NL writer
#

import os
from os.path import abspath, dirname, join
currdir = dirname(abspath(__file__))

import pyutilib.th as unittest

from pyomo.opt import ProblemFormat
from pyomo.core import *

class TestSuffix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pyomo.environ

    # test that EXPORT suffixes on variables,
    # constraints, objectives, and models
    # will end up in the NL file with integer tags
    def test_EXPORT_suffixes_int(self):
        model = ConcreteModel()
        model.junk = Suffix(direction=Suffix.EXPORT,datatype=Suffix.INT)
        model.junk_inactive = Suffix(direction=Suffix.EXPORT,datatype=Suffix.INT)

        model.x = Var()
        model.junk.set_value(model.x,1)
        model.junk_inactive.set_value(model.x,1)

        model.y = Var([1,2], dense=True)
        model.junk.set_value(model.y,2)
        model.junk_inactive.set_value(model.y,2)

        model.obj = Objective(expr=model.x+summation(model.y))
        model.junk.set_value(model.obj,3)
        model.junk_inactive.set_value(model.obj,3)

        model.conx = Constraint(expr=model.x>=1)
        model.junk.set_value(model.conx,4)
        model.junk_inactive.set_value(model.conx,4)

        model.cony = Constraint([1,2],rule=lambda model,i: model.y[i]>=1)
        model.junk.set_value(model.cony,5)
        model.junk_inactive.set_value(model.cony,5)

        model.junk.set_value(model,6)
        model.junk_inactive.set_value(model,6)

        # This one should NOT end up in the NL file
        model.junk_inactive.deactivate()

        model.write(filename=join(currdir,"EXPORT_suffixes.test.nl"),
                    format=ProblemFormat.nl,
                    io_options={"symbolic_solver_labels" : False})

        self.assertFileEqualsBaseline(join(currdir,"EXPORT_suffixes.test.nl"),
                                      join(currdir,"EXPORT_suffixes_int.baseline.nl"))

    # test that EXPORT suffixes on variables,
    # constraints, objectives, and models
    # will end up in the NL file with floating point tags
    def test_EXPORT_suffixes_float(self):
        model = ConcreteModel()
        model.junk = Suffix(direction=Suffix.EXPORT,datatype=Suffix.FLOAT)
        model.junk_inactive = Suffix(direction=Suffix.EXPORT,datatype=Suffix.FLOAT)

        model.x = Var()
        model.junk.set_value(model.x,1)
        model.junk_inactive.set_value(model.x,1)

        model.y = Var([1,2], dense=True)
        model.junk.set_value(model.y,2)
        model.junk_inactive.set_value(model.y,2)

        model.obj = Objective(expr=model.x+summation(model.y))
        model.junk.set_value(model.obj,3)
        model.junk_inactive.set_value(model.obj,3)

        model.conx = Constraint(expr=model.x>=1)
        model.junk.set_value(model.conx,4)
        model.junk_inactive.set_value(model.conx,4)

        model.cony = Constraint([1,2],rule=lambda model,i: model.y[i]>=1)
        model.junk.set_value(model.cony,5)
        model.junk_inactive.set_value(model.cony,5)

        model.junk.set_value(model,6)
        model.junk_inactive.set_value(model,6)

        # This one should NOT end up in the NL file
        model.junk_inactive.deactivate()

        model.write(filename=join(currdir,"EXPORT_suffixes.test.nl"),
                    format=ProblemFormat.nl,
                    io_options={"symbolic_solver_labels" : False})

        self.assertFileEqualsBaseline(join(currdir,"EXPORT_suffixes.test.nl"),
                                      join(currdir,"EXPORT_suffixes_float.baseline.nl"))

    # Test that user defined ref suffixes fail to
    # merge with those created from translating the SOSConstraint
    # component when variables get assigned duplicate values for ref
    def test_EXPORT_suffixes_with_SOSConstraint_duplicateref(self):
        model = ConcreteModel()
        model.ref = Suffix(direction=Suffix.EXPORT,datatype=Suffix.INT)
        model.y = Var([1,2,3])
        model.obj = Objective(expr=summation(model.y))

        # The NL writer will convert this constraint to ref and sosno
        # suffixes on model.y
        model.sos_con = SOSConstraint(var=model.y, index=[1,2,3], sos=1)

        for i,val in zip([1,2,3],[11,12,13]):
            model.ref.set_value(model.y[i],val)
        
        try:
            model.write(filename=join(currdir,"junk.nl"),
                        format=ProblemFormat.nl,
                        io_options={"symbolic_solver_labels" : False})
        except RuntimeError:
            pass
        else:
            os.remove(join(currdir,"junk.nl"))
            self.fail("The NL writer should have thrown an exception "\
                      "when overlap of SOSConstraint generated suffixes "\
                      "and user declared suffixes occurs.")
        
        try:
            os.remove(join(currdir,"junk.nl"))
        except:
            pass

    # Test that user defined sosno suffixes fail to
    # merge with those created from translating the SOSConstraint
    # component when variables get assigned duplicate values for sosno
    def test_EXPORT_suffixes_with_SOSConstraint_duplicatesosno(self):
        model = ConcreteModel()
        model.sosno = Suffix(direction=Suffix.EXPORT,datatype=Suffix.INT)
        model.y = Var([1,2,3])
        model.obj = Objective(expr=summation(model.y))

        # The NL writer will convert this constraint to ref and sosno
        # suffixes on model.y
        model.sos_con = SOSConstraint(var=model.y, index=[1,2,3], sos=1)

        for i in [1,2,3]:
            model.sosno.set_value(model.y[i],-1)
        
        try:
            model.write(filename=join(currdir,"junk.nl"),
                        format=ProblemFormat.nl,
                        io_options={"symbolic_solver_labels" : False})
        except RuntimeError:
            pass
        else:
            os.remove(join(currdir,"junk.nl"))
            self.fail("The NL writer should have thrown an exception "\
                      "when overlap of SOSConstraint generated suffixes "\
                      "and user declared suffixes occurs.")
        try:
            os.remove(join(currdir,"junk.nl"))
        except:
            pass

    # Test that user defined sosno suffixes fail to
    # merge with those created from translating the SOSConstraint
    # component when variables get assigned duplicate values for sosno
    def test_EXPORT_suffixes_no_datatype(self):
        model = ConcreteModel()
        model.sosno = Suffix(direction=Suffix.EXPORT,datatype=None)
        model.y = Var([1,2,3])
        model.obj = Objective(expr=summation(model.y))

        # The NL writer will convert this constraint to ref and sosno
        # suffixes on model.y
        model.sos_con = SOSConstraint(var=model.y, index=[1,2,3], sos=1)

        for i in [1,2,3]:
            model.sosno.set_value(model.y[i],-1)
        
        try:
            model.write(filename=join(currdir,"junk.nl"),
                        format=ProblemFormat.nl,
                        io_options={"symbolic_solver_labels" : False})
        except RuntimeError:
            pass
        else:
            os.remove(join(currdir,"junk.nl"))
            self.fail("The NL writer should have thrown an exception "\
                      "when using an EXPORT suffix with datatype=None")
        try:
            os.remove(join(currdir,"junk.nl"))
        except:
            pass

if __name__ == "__main__":
    unittest.main()
