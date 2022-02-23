#################################################################################################################
# Copyright (c) 2022
# *** Copyright Notice ***
# “SOFTWARE NAME” was produced under the DOE Carbon Capture Simulation Initiative (CCSI), and is
# copyright (c) 2022 by the software owners: TRIAD, LLNS, BERKELEY LAB, PNNL, UT-Battelle, LLC, NOTRE
# DAME, PITT, UT Austin, TOLEDO, WVU, et al. All rights reserved.
# 
# NOTICE. This Software was developed under funding from the U.S. Department of Energy and the U.S.
# Government consequently retains certain rights. As such, the U.S. Government has been granted for itself
# and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to
# reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display
# publicly, and to permit other to do so.
# 
# *** License Agreement ***
# 
# “SOFTWARE NAME” Copyright (c) 2022, by the software owners: TRIAD, LLNS, BERKELEY LAB, PNNL, UT-
# Battelle, LLC, NOTRE DAME, PITT, UT Austin, TOLEDO, WVU, et al. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted provided
# that the following conditions are met:
# (1) Redistributions of source code must retain the above copyright notice, this list of conditions and the
# following disclaimer.
# (2) Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
# the following disclaimer in the documentation and/or other materials provided with the distribution.
# (3) Neither the name of the Carbon Capture Simulation for Industry Impact, TRIAD, LLNS, BERKELEY LAB,
# PNNL, UT-Battelle, LLC, ORNL, NOTRE DAME, PITT, UT Austin, TOLEDO, WVU, U.S. Dept. of Energy nor
# the names of its contributors may be used to endorse or promote products derived from this software
# without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
# THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# You are under no obligation whatsoever to provide any bug fixes, patches, or upgrades to the features,
# functionality or performance of the source code ("Enhancements") to anyone; however, if you choose to
# make your Enhancements available either publicly, or directly to Lawrence Berkeley National Laboratory,
# without imposing a separate written license agreement for such Enhancements, then you hereby grant
# the following license: a non-exclusive, royalty-free perpetual license to install, use, modify, prepare
# derivative works, incorporate into other computer software, distribute, and sublicense such
# enhancements or derivative works thereof, in binary and source code form.
# 
#################################################################################################################

from pyomo.environ import *
from pyomo.dae import *
import numpy as np
import pandas as pd 
from itertools import permutations, product, combinations

import pyomo.common.unittest as unittest
from fim_doe import *

class TestMeasurement(unittest.TestCase):
    '''Test the measurement class
    '''
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
    ''' Test the parameter class 
    '''
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