#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.dependencies import (
    numpy as np, numpy_available,
    pandas as pd, pandas_available,
    scipy, scipy_available,
    matplotlib, matplotlib_available,
)
imports_present = numpy_available & pandas_available & scipy_available

uuid_available = True
try:
    import uuid
except:
    uuid_available = False

import pyutilib.th as unittest
import os
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.scenariocreator as sc
import pyomo.contrib.parmest.examples.semibatch.scencreate as sbc
import pyomo.environ as pyo
from pyomo.environ import SolverFactory
ipopt_available = SolverFactory('ipopt').available()

testdir = os.path.dirname(os.path.abspath(__file__))


@unittest.skipIf(not imports_present, "Cannot test parmest: required dependencies are missing")
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class pamest_Scenario_creator_reactor_design(unittest.TestCase):
    
    def setUp(self):
        from pyomo.contrib.parmest.examples.reactor_design.reactor_design import reactor_design_model
          
        # Data from the design 
        data = pd.DataFrame(data=[[1.05, 10000, 3458.4, 1060.8, 1683.9, 1898.5],
                                  [1.10, 10000, 3535.1, 1064.8, 1613.3, 1893.4],
                                  [1.15, 10000, 3609.1, 1067.8, 1547.5, 1887.8],
                                  [1.20, 10000, 3680.7, 1070.0, 1486.1, 1881.6],
                                  [1.25, 10000, 3750.0, 1071.4, 1428.6, 1875.0],
                                  [1.30, 10000, 3817.1, 1072.2, 1374.6, 1868.0],
                                  [1.35, 10000, 3882.2, 1072.4, 1324.0, 1860.7],
                                  [1.40, 10000, 3945.4, 1072.1, 1276.3, 1853.1],
                                  [1.45, 10000, 4006.7, 1071.3, 1231.4, 1845.3],
                                  [1.50, 10000, 4066.4, 1070.1, 1189.0, 1837.3],
                                  [1.55, 10000, 4124.4, 1068.5, 1148.9, 1829.1],
                                  [1.60, 10000, 4180.9, 1066.5, 1111.0, 1820.8],
                                  [1.65, 10000, 4235.9, 1064.3, 1075.0, 1812.4],
                                  [1.70, 10000, 4289.5, 1061.8, 1040.9, 1803.9],
                                  [1.75, 10000, 4341.8, 1059.0, 1008.5, 1795.3],
                                  [1.80, 10000, 4392.8, 1056.0,  977.7, 1786.7],
                                  [1.85, 10000, 4442.6, 1052.8,  948.4, 1778.1],
                                  [1.90, 10000, 4491.3, 1049.4,  920.5, 1769.4],
                                  [1.95, 10000, 4538.8, 1045.8,  893.9, 1760.8]], 
                          columns=['sv', 'caf', 'ca', 'cb', 'cc', 'cd'])

        theta_names = ['k1', 'k2', 'k3']
        
        def SSE(model, data): 
            expr = (float(data['ca']) - model.ca)**2 + \
                   (float(data['cb']) - model.cb)**2 + \
                   (float(data['cc']) - model.cc)**2 + \
                   (float(data['cd']) - model.cd)**2
            return expr
        
        self.pest = parmest.Estimator(reactor_design_model, data, theta_names, SSE)

    def test_scen_from_exps(self):
        scenmaker = sc.ScenarioCreator(self.pest, "ipopt")
        experimentscens = sc.ScenarioSet("Experiments")
        scenmaker.ScenariosFromExperiments(experimentscens)
        experimentscens.write_csv("delme_exp_csv.csv")
        df = pd.read_csv("delme_exp_csv.csv")
        os.remove("delme_exp_csv.csv")
        # March '20: all reactor_design experiments have the same theta values!
        k1val = df.loc[5].at["k1"] 
        self.assertAlmostEqual(k1val, 5.0/6.0, places=2)
        tval = experimentscens.ScenarioNumber(0).ThetaVals["k1"]
        self.assertAlmostEqual(tval, 5.0/6.0, places=2)

        
    @unittest.skipIf(not uuid_available, "The uuid module is not available")
    def test_no_csv_if_empty(self):
        # low level test of scenario sets
        # verify that nothing is written, but no errors with empty set

        emptyset = sc.ScenarioSet("empty")
        tfile = uuid.uuid4().hex+".csv"
        emptyset.write_csv(tfile)
        self.assertFalse(os.path.exists(tfile),
                         "ScenarioSet wrote csv in spite of empty set")

        


@unittest.skipIf(not imports_present, "Cannot test parmest: required dependencies are missing")
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class  pamest_Scenario_creator_semibatch(unittest.TestCase):
    
    def setUp(self):
        import pyomo.contrib.parmest.examples.semibatch.semibatch as sb
        import json

        # Vars to estimate in parmest
        theta_names = ['k1', 'k2', 'E1', 'E2']

        self.fbase = os.path.join(testdir,"..","examples","semibatch")
        # Data, list of dictionaries
        data = [] 
        for exp_num in range(10):
            fname = "exp"+str(exp_num+1)+".out"
            fullname = os.path.join(self.fbase, fname)
            with open(fullname,'r') as infile:
                d = json.load(infile)
                data.append(d)

        # Note, the model already includes a 'SecondStageCost' expression 
        # for the sum of squared error that will be used in parameter estimation

        self.pest = parmest.Estimator(sb.generate_model, data, theta_names)
        

    def test_semibatch_bootstrap(self):

        scenmaker = sc.ScenarioCreator(self.pest, "ipopt")
        bootscens = sc.ScenarioSet("Bootstrap")
        numtomake = 2
        scenmaker.ScenariosFromBoostrap(bootscens, numtomake, seed=1134)
        tval = bootscens.ScenarioNumber(0).ThetaVals["k1"]
        self.assertAlmostEqual(tval, 20.64, places=1)

    def test_semibatch_example(self):
        # this is referenced in the documentation so at least look for smoke
        sbc.main(self.fbase)
        
if __name__ == '__main__':
    unittest.main()
