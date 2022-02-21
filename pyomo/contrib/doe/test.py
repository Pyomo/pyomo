from pyomo.environ import *
from pyomo.dae import *
import numpy as np
import pandas as pd 
from itertools import permutations, product, combinations

import pyomo.common.unittest as unittest
from fim_doe import *

class TestMeasurement(unittest.TestCase):
    
    def test_setup(self):
        # generate a set of measurements with extra index CA, CB, CC
        # each extra index has different measure time points
        t_measure_ca = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
        t_measure_cb = [0, 0.25, 0.5, 0.75, 1]
        t_measure_cc = [0, 0.125, 0.375, 0.625, 0.875, 1]
        var = {'C':{'CA': 5, 'CB': 2, 'CC': 1}}
        measure_pass = {'C':{"CA": t_measure_ca, "CB": t_measure_cb, "CC": t_measure_cc}}

        measure_class = Measurements(measure_pass, variance=var)
        
        # test names, variance, time sets
        self.assertEqual(measure_class.flatten_measure_name, ['C_index_CA', 'C_index_CB', 'C_index_CC'])
        self.assertEqual(measure_class.flatten_variance, {'C_index_CA': 5, 'C_index_CB': 2, 'C_index_CC': 1})
        self.assertEqual(measure_class.flatten_measure_timeset['C_index_CB'], [0, 0.25, 0.5, 0.75, 1])
        
        # test subset feature
        subset_1 = {'C':{'CA': t_measure_cb}}
        measure_subclass1 = Measurements(subset_1)
        self.assertTrue(measure_class.check_subset(measure_subclass1))
        

class TestParameter(unittest.TestCase):
    
    def test_setup(self):
        # set up parameter class
        param_dict = {'A1': 84.79, 'A2': 371.72, 'E1': 7.78, 'E2': 15.05}

        scenario_gene = Scenario_generator(param_dict, formula='central', step=0.1)
        parameter_set = scenario_gene.simultaneous_scenario()
    
        self.assertAlmostEqual(parameter_set['A1'][0], 93.269, places=1)
        self.assertAlmostEqual(parameter_set['A1'][4], 76.311, places=1)
        self.assertEqual(parameter_set['jac-index']['A2'], [1,5])
        self.assertEqual(parameter_set['scena-name'], [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertAlmostEqual(parameter_set['eps-abs']['E1'], 1.556, places=1)
        
        scenario_gene.generate_sequential_para()
        self.assertEqual(scenario_gene.scenario_para['A2'], [1,5])
        self.assertAlmostEqual(scenario_gene.eps_abs['A1'], 16.958, places=1)
        self.assertAlmostEqual(scenario_gene.next_sequential_scenario(2)['E1'][0], 8.558, places=1)
        self.assertEqual(scenario_gene.next_sequential_scenario(2)['scena-name'], [0])
        
if __name__ == '__main__':
    unittest.main()