#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#
#  Pyomo.DoE was produced under the Department of Energy Carbon Capture Simulation 
#  Initiative (CCSI), and is copyright (c) 2022 by the software owners: 
#  TRIAD National Security, LLC., Lawrence Livermore National Security, LLC., 
#  Lawrence Berkeley National Laboratory, Pacific Northwest National Laboratory,  
#  Battelle Memorial Institute, University of Notre Dame,
#  The University of Pittsburgh, The University of Texas at Austin, 
#  University of Toledo, West Virginia University, et al. All rights reserved.
# 
#  NOTICE. This Software was developed under funding from the 
#  U.S. Department of Energy and the U.S. Government consequently retains 
#  certain rights. As such, the U.S. Government has been granted for itself
#  and others acting on its behalf a paid-up, nonexclusive, irrevocable, 
#  worldwide license in the Software to reproduce, distribute copies to the 
#  public, prepare derivative works, and perform publicly and display
#  publicly, and to permit other to do so.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
from pyomo.contrib.doe import Measurements, DesignVariables, ScenarioGenerator, finite_difference_lib

class TestMeasurement(unittest.TestCase):
    """Test the Measurements class, specify, add_element, update_variance, check_subset functions.
    """
    def test_setup(self):
        ### add_element function 
        
        # control time for C [h]
        t_control = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
        # control time for T [h]
        t_control2 = [0.2,0.4,0.6,0.8]

        # set up measurements 
        total_name = ["C", "T"]
        extra_index = [[["CA", "CB", "CC"]], [[1,3,5]]]
        time_index = [t_control, t_control2]  
        
        measure_class = Measurements()
        measure_class.add_elements(total_name, extra_index=extra_index, time_index = time_index)
        
        # test names, variance 
        self.assertEqual(measure_class.name[0], 'C[CA,0]')
        self.assertEqual(measure_class.name[1], 'C[CA,0.125]')
        self.assertEqual(measure_class.name[-1], 'T[5,0.8]')
        self.assertEqual(measure_class.name[-2], 'T[5,0.6]')
        self.assertEqual(measure_class.variance['T[5,0.4]'], 1)
        self.assertEqual(measure_class.variance['T[5,0.6]'], 1)


        ### update_variance 
        new_var = {'C[CA,0]': 1, 'C[CA,0.125]': 1, 'C[CA,0.25]': 1, 'C[CA,0.375]': 1, 'C[CA,0.5]': 1, 
                   'C[CA,0.625]': 1, 'C[CA,0.75]': 1, 'C[CA,0.875]': 1, 'C[CA,1]': 1, 'C[CB,0]': 1, 
                   'C[CB,0.125]': 1, 'C[CB,0.25]': 1, 'C[CB,0.375]': 1, 'C[CB,0.5]': 1, 'C[CB,0.625]': 1, 
                   'C[CB,0.75]': 1, 'C[CB,0.875]': 1, 'C[CB,1]': 1, 'C[CC,0]': 1, 'C[CC,0.125]': 1, 
                   'C[CC,0.25]': 1, 'C[CC,0.375]': 1, 'C[CC,0.5]': 1, 'C[CC,0.625]': 1, 'C[CC,0.75]': 1, 
                   'C[CC,0.875]': 1, 'C[CC,1]': 1, 'T[1,0.2]': 1, 'T[1,0.4]': 1, 'T[1,0.6]': 1, 
                   'T[1,0.8]': 1, 'T[3,0.2]': 1, 'T[3,0.4]': 1, 'T[3,0.6]': 1, 'T[3,0.8]': 1, 'T[5,0.2]': 1,
                     'T[5,0.4]': 10, 'T[5,0.6]': 12, 'T[5,0.8]': 1}
        measure_class.update_variance(new_var)

        self.assertEqual(measure_class.variance['T[5,0.4]'], 10)
        self.assertEqual(measure_class.variance['T[5,0.6]'], 12)

        ### specify function 
        var_names = ['C[CA,0]', 'C[CA,0.125]', 'C[CA,0.875]', 'C[CA,1]', 
                     'C[CB,0]', 'C[CB,0.125]', 'C[CB,0.25]', 'C[CB,0.375]', 
                     'C[CC,0]', 'C[CC,0.125]', 'C[CC,0.25]', 'C[CC,0.375]']

        measure_class2 = Measurements()
        measure_class2.specify(var_names)

        self.assertEqual(measure_class2.name[1], 'C[CA,0.125]')
        self.assertEqual(measure_class2.name[-1], 'C[CC,0.375]')

        ### check_subset function 
        self.assertTrue(measure_class.check_subset(measure_class2))
        

class TestDesignVariable(unittest.TestCase):
    """Test the DesignVariable class, specify, add_element, add_bounds, update_values.
    """
    def test_setup(self):
        ### add_element function 
        total_name = ["CA0", "T"]
        # control time for C [h]
        t_control = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
        dtime_index = [[0], t_control] 
        exp1 = [5, 570, 300, 300, 300, 300, 300, 300, 300, 300]
        upper_bound = [5, 700, 700, 700, 700, 700, 700, 700, 700, 700]
        lower_bound = [1, 300, 300, 300, 300, 300, 300, 300, 300, 300]

        design_gen = DesignVariables()
        design_gen.add_elements(total_name, time_index = dtime_index, values=exp1, 
                                upper_bound=upper_bound, lower_bound=lower_bound)

        self.assertEqual(design_gen.name, ['CA0[0]', 'T[0]', 'T[0.125]', 'T[0.25]', 'T[0.375]',
                                                   'T[0.5]', 'T[0.625]', 'T[0.75]', 'T[0.875]', 'T[1]'])
        self.assertEqual(design_gen.special_set_value['CA0[0]'], 5)
        self.assertEqual(design_gen.special_set_value['T[0]'], 570)
        self.assertEqual(design_gen.upper_bound['CA0[0]'], 5)
        self.assertEqual(design_gen.upper_bound['T[0]'], 700)
        self.assertEqual(design_gen.lower_bound['CA0[0]'], 1)
        self.assertEqual(design_gen.lower_bound['T[0]'], 300)

        exp1 = [4, 600, 300, 300, 300, 300, 300, 300, 300, 300]
        design_gen.update_values(exp1)
        self.assertEqual(design_gen.special_set_value['CA0[0]'], 4)
        self.assertEqual(design_gen.special_set_value['T[0]'], 600)


class TestParameter(unittest.TestCase):
    """ Test the ScenarioGenerator class, generate_scenario function.
    """
    def test_setup(self):
        # set up parameter class
        param_dict = {'A1': 84.79, 'A2': 371.72, 'E1': 7.78, 'E2': 15.05}

        scenario_gene = ScenarioGenerator(param_dict, formula=finite_difference_lib.central, step=0.1)
        parameter_set = scenario_gene.generate_scenario()
    
        self.assertAlmostEqual(parameter_set['eps-abs']['A1'], 16.9582, places=1)
        self.assertAlmostEqual(parameter_set['eps-abs']['E1'], 1.5554, places=1)
        self.assertEqual(parameter_set['scena_num']['A2'], [2,3])
        self.assertEqual(parameter_set['scena_num']['E1'], [4,5])
        self.assertAlmostEqual(parameter_set['scenario'][0]['A1'], 93.2699, places=1)
        self.assertAlmostEqual(parameter_set['scenario'][2]['A2'], 408.8895, places=1)
        self.assertAlmostEqual(parameter_set['scenario'][-1]['E2'], 13.54, places=1)
        self.assertAlmostEqual(parameter_set['scenario'][-2]['E2'], 16.55, places=1)

        
if __name__ == '__main__':
    unittest.main()
