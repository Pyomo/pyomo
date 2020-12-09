#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#

import six
import pickle
import base64
import ast
import os
from os.path import abspath, dirname

from pyutilib.pyro import using_pyro4
import pyutilib.th as unittest

from pyomo.common.collections import Options
from pyomo.common.tempfiles import TempfileManager
import pyomo.opt
from pyomo.environ import (ConcreteModel, RangeSet, Var,
                           Objective, Constraint, sum_product)
import pyomo.scripting.pyro_mip_server


currdir = dirname(abspath(__file__))+os.sep

solvers = pyomo.opt.check_available_solvers('glpk')

class TestWorker(pyomo.scripting.pyro_mip_server.PyomoMIPWorker):

    def __init__(self):
        self._verbose = True
        self._current_task_client = None

class Test(unittest.TestCase):

    def setUp(self):
        self.worker = TestWorker()

    def tearDown(self):
        TempfileManager.clear_tempfiles()
        del self.worker

    @unittest.skipIf(not 'glpk' in solvers, "glpk solver is not available")
    def test_t1(self):
        # Run a simple model
        model = ConcreteModel()
        model.A = RangeSet(1,4)
        model.x = Var(model.A, bounds=(-1,1))
        def obj_rule(model):
            return sum_product(model.x)
        model.obj = Objective(rule=obj_rule)
        def c_rule(model):
            expr = 0
            for i in model.A:
                expr += i*model.x[i]
            return expr == 0
        model.c = Constraint(rule=c_rule)

        #
        data = Options()
        data.suffixes = {}
        data.solver_options = {}
        data.warmstart_filename = None
        data.filename = currdir+'t1.lp'
        model.write(data['filename'])
        INPUT = open(data['filename'],'r')
        data['file'] = INPUT.read()
        INPUT.close()
        data['opt'] = 'glpk'
        data.kwds = {}
        #
        results = self.worker.process(data)

        # Decode, evaluate and unpickle results
        if using_pyro4:
            # These two conversions are in place to unwrap
            # the hacks placed in the pyro_mip_server
            # before transmitting the results
            # object. These hacks are put in place to
            # avoid errors when transmitting the pickled
            # form of the results object with the default Pyro4
            # serializer (Serpent)
            if six.PY3:
                results = base64.decodebytes(
                    ast.literal_eval(results))
            else:
                results = base64.decodestring(results)

        results = pickle.loads(results)

        #
        results.write(filename=currdir+"t1.out", format='json')
        self.assertMatchesJsonBaseline(currdir+"t1.out",currdir+"t1.txt", tolerance=1e-4)
        self.assertEqual(results._smap_id, None)
        os.remove(data['filename'])


if __name__ == "__main__":
    unittest.main()

