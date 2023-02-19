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

import warnings
from specialSet import SpecialSet

class Measurements(SpecialSet):
    def __init__(self, variance=None):
        """
        This class stores information on which algebraic and differential variables in the Pyomo model are considered measurements. 
        """
        super().__init__()
        self._generate_variance(variance)

    def specify(self, self_define_res):

        self.measurement_name = super().specify(self_define_res)

    def add_elements(self, var_name, extra_index=None, time_index=[0]):
        """

        Parameters
        -----------
        var_name: a ``list`` of measurement var names 
            extra_index: a ``list`` containing extra indexes except for time indexes 
                if default (None), no extra indexes needed for all var in var_name
                if it is a nested list, it is a ``list`` of ``list`` of ``list``, 
                they are different multiple sets of indexes for different var_name
                for e.g., extra_index[0] are all indexes for var_name[0], extra_index[0][0] are the first index for var_name[0]
            time_index: a ``list`` containing time indexes
                default choice is [0], means this is an algebraic variable
                if it is a nested list, it is a ``list`` of ``lists``, they are different time set for different var in var_name
        """
        self.measurement_name =  super().add_elements(var_name=var_name, extra_index=extra_index, time_index=time_index)

    def _generate_variance(self, variance):
        return 

    def _check_names(self):
        """
        Check if the measurement information provided are valid to use. 
        """

        assert type(self.measure_varname)==dict, "Measurement_var should be a dictionary."

        for name in self.measure_varname:
            if len(self.measure_varname[name]) != 2: 
                warnings.warn("There should be two lists for each measurement. This warning indicates a potential modeling error.")

            if self.measure_varname[name][0]:
                for i in self.measure_varname[name][0]:
                    print(i)
                    if type(i) is not list:
                        warnings.warn("Each extra index should form a seperate list. This warning indicates a potential modeling error.")

            if self.measure_varname[name][1]:
                for i in self.measure_varname[name][1]:
                    if type(i) is list:
                        warnings.warn("Each element in time index should be a number.")