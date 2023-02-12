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
import itertools

class Measurements:
    def __init__(self, self_define_res=None,  measurement_var=None, variance=None, ind_string='_index_'):
        """
        This class stores information on which algebraic and differential variables in the Pyomo model are considered measurements. 
        This includes the functionality to specify indices for these measurement variables. 
        For example, with a partial differential algebraic equation model, 
        these measurement index sets can specify which spatial and temporal coordinates each measurement is available. 
        Moreover, this class supports defining the covariance matrix for all measurements.

        Parameters
        ----------
        self_define_res: a ``list`` of ``string``, containing the measurement variable names with indexs, 
            for e.g. "C['CA', 23, 0]". 
            If this is defined, no need to define ``measurement_var``, ``extra_idx``, ``time_idx``.
        measurement_var: if not given self_define_res, this needs to be given to provide measurement names.
             a ``dict``. Keys are measurement Var name, values are a list of two lists. 
             The first list contains extra index. 
                extra_idx: a ``list`` of ``list``. Each list contains an extra index for the measurement variables. 
                For e.g., [['CA', 'CB', 'CC'], [10, 23, 28]] means there are two extra indexes for measurement variables
                besides the time index.
            The second list contains time index. 
                time_idx: a ``list`` containing timepoints. 

            For e.g., {'C':[['CA', 'CB', 'CC'], [10, 23, 28]], [1,2,3]} is a valid input. The measurement names will be 
            from "C['CA', 10, 1]" to "C['CB',28,3]".

        variance:
            a ``dict``, keys are measurement variable names, values are a dictionary, keys are its extra index,
            values are its variance (a scalar number), values are its variance if there is no extra index for this measurement.
            For e.g., for the kinetics illustrative example, it should be {'C':{'CA': 10, 'CB': 1, 'CC': 2}}.
            If given None, the default is {'C':{'CA': 1, 'CB': 1, 'CC': 1}}.
        ind_string:
            a ''string'', used to flatten the name of variables and extra index. Default is '_index_'.
            For e.g., for {'C':{'CA': 10, 'CB': 1, 'CC': 2}}, the reformulated name is 'C_index_CA'.
        """
        if self_define_res:
            self.measurement_name = self_define_res
        elif measurement_var:
            self.measure_varname = measurement_var
            self._check_names()
            self._measure_name()
        else:
            raise AttributeError("No measurement names provided. Define either self_define_res or measurement_var.")

    def _measure_name(self):
        """Return pyomo string name
        """
        names = list(self.measure_varname.keys())
        name_list = []
        for n in names:
            name_data = str(n)

            # first flatten all indexes into a list 
            all_index_list = [] # contains all index lists
            if self.measure_varname[n][0]:
                for index_list in self.measure_varname[n][0]:
                    all_index_list.append(index_list)
            if self.measure_varname[n][1]:
                all_index_list.append(self.measure_varname[n][1])

            if not all_index_list: # there is no any index for this measurement 
                name_list.append(name_data)

            else: # add indexes to the variable name 

                # all idnex list for one variable, such as ["CA", 10, 1]
                all_index_for_var = list(itertools.product(*all_index_list))

                for lst in all_index_for_var:
                    name1 = name_data+"["
                    for i, idx in enumerate(lst):
                        #if idx is str:
                        #    name1 += "'" + str(idx)+"'"
                        #else:
                        name1 += str(idx)

                        if i==len(lst)-1:
                            name1 += "]"
                        else:
                            name1 += ","

                    name_list.append(name1)
        print(name_list)

        self.measurement_name = name_list
        

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