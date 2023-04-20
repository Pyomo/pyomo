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

import itertools

class VariablesWithIndices: 
    def __init__(self):
        """This class provides utility methods for DesignVariables and MeasurementVariables to create
        lists of Pyomo variable names with an arbitrary number of indices. 
        """
        self.variable_names = []

    def set_variable_name_list(self, self_define_res):
        """
        Used for user to provide defined string names

        Parameter
        ---------
        self_define_res: a ``list`` of ``string``, containing the variable names with indexs, 
            for e.g. "C['CA', 23, 0]".
        """
        self.variable_names = self_define_res

        return self.variable_names

    def add_variables(self, var_name, indices=None, time_index_position=None, values=None, 
                     lower_bounds=None, upper_bounds=None):
        """
        Used for generating string names with indexes. 

        Parameter 
        ---------
        var_name: variable name in ``string`` 
        indices: a ``dict`` containing indexes 
            if default (None), no extra indexes needed for all var in var_name
            for e.g., {0:["CA", "CB", "CC"], 1: [1,2,3]}. 
        time_index_position: an integer indicates which index is the time index  
            for e.g., 1 is the time index position in the indices example. 
        values: a ``list`` containing values which has the same shape of flattened variables 
            default choice is None, means there is no give nvalues 
        lower_bounds: a ``list `` of lower bounds. If given a scalar number, it is set as the lower bounds for all variables.
        upper_bounds: a ``list`` of upper bounds. If given a scalar number, it is set as the upper bounds for all variables.
        
        Return
        ------
        if not defining values, return a set of variable names 
        if defining values, return a dictionary of variable names and its value 
        """
        num_indices = self._add_variables(var_name, indices=indices, time_index_position=time_index_position)

        self._names(num_indices, indices, time_index_position, values, lower_bounds, upper_bounds)

        if values:
            # this dictionary keys are special set, values are its value
            self.variable_names_value = self._generate_dict(values)

        if lower_bounds:
            if type(lower_bounds) in [int, float]:
                lower_bounds = [lower_bounds]*len(self.variable_names)
            self.lower_bounds = self._generate_dict(lower_bounds)
        
        if upper_bounds:
            if type(upper_bounds) in [int, float]:
                upper_bounds = [upper_bounds]*len(self.variable_names)
            self.upper_bounds = self._generate_dict(upper_bounds)

        return self.variable_names
    
    def update_values(self, values):
        """
        Used for updating values

        Parameters
        ---------
         values: a ``list`` containing values which has the same shape of flattened variables 
            default choice is None, means there is no give nvalues 
        """
        self.variable_names_value = self._generate_dict(values)
        

    def _generate_dict(self, values):
        """
        Given a list of values, return a dictionary, keys are special set names. 
        """
        value_map = {}
        for i in range(len(values)):
            value_map[self.variable_names[i]] = values[i]

        return value_map


    def _add_variables(self, var_name, indices=None, time_index_position=None):
        """
        Used for generating string names with indexes. 

        Parameter 
        ---------
        var_name: a ``list`` of var names 
        indices: a ``dict`` containing indexes 
            if default (None), no extra indexes needed for all var in var_name
            for e.g., {0:["CA", "CB", "CC"], 1: [1,2,3]}. 
        time_index_position: an integer indicates which index is the time index  
            for e.g., 1 is the time index position in the indices example. 
        """
        # first combine all indexes into a list 
        all_index_list = [] # contains all index lists
        if indices:
            for index_pointer in indices: 
                all_index_list.append(indices[index_pointer])

        # all idnex list for one variable, such as ["CA", 10, 1]
        all_index_for_var = list(itertools.product(*all_index_list))

        for lst in all_index_for_var:
            name1 = var_name+"["
            for i, idx in enumerate(lst):
                name1 += str(idx)

                # if i is the last index, close the []. if not, add a "," for the next index. 
                if i==len(lst)-1:
                    name1 += "]"
                else:
                    name1 += ","

            self.variable_names.append(name1)

        return len(all_index_for_var)


    def _names(self, len_indices, indices, time_index_position, values, lower_bounds, upper_bounds):
        """
        Check if the measurement information provided are valid to use. 
        """
        
        if time_index_position not in indices:
            raise ValueError("time index cannot be found in indices.")

        # if given a list, check if bounds have the same length with flattened variable 
        if values and len(values) != len_indices:
            raise ValueError("Values is of different length with indices.")
        
        if lower_bounds and type(lower_bounds)==list and len(lower_bounds)!= len_indices:
            raise ValueError("Lowerbounds is of different length with indices.")
        
        if upper_bounds and type(upper_bounds)==list and len(upper_bounds)!= len_indices:
            raise ValueError("Upperbounds is of different length with indices.")
        
        

    
class MeasurementVariables(VariablesWithIndices):
    def __init__(self):
        """
        This class stores information on which algebraic and differential variables in the Pyomo model are considered measurements. 
        """
        super().__init__()

    def specify(self, self_define_res):
        """
        User can pass already defined measurement names here
        """
        self.name = super().specify(self_define_res)

        # generate default variance
        self._use_identity_variance()

    def add_variables(self, var_name, indices=None, time_index_position=None):
        """
        Parameters 
        -----------
        var_name: a ``list`` of var names 
        indices: a ``dict`` containing indexes 
            if default (None), no extra indexes needed for all var in var_name
            for e.g., {0:["CA", "CB", "CC"], 1: [1,2,3]}. 
        time_index_position: an integer indicates which index is the time index  
            for e.g., 1 is the time index position in the indices example. 
        """
        self.name =  super().add_variables(var_name=var_name, indices=indices, time_index_position=time_index_position)

        # generate default variance
        self._use_identity_variance()

    def update_variance(self, variance):
        """If not using default variance 

        Parameters 
        ----------
        variance: 
            a ``dict``, keys are measurement variable names, values are its variance (a scalar number)
            For e.g., for the kinetics example, it should be {'CA[0]':10, 'CA[0.125]': 1, ...., 'CC[1]': 2}. 
            If given None, the default is {'CA[0]':1, 'CA[0.125]': 1, ...., 'CC[1]': 1}
        """
        self.variance = variance 

    def check_subset(self, subset_class):
        """
        Check if subset_class is a subset of the current measurement object
        
        Parameters
        ----------
        subset_class: a measurement object
        """
        for nam in subset_class.name:
            if nam not in self.name:
                raise ValueError("Measurement not in the set: ", nam)
        
        return True
        
    def _use_identity_variance(self):
        """Generate the variance dictionary. 
        """
        self.variance = {}
        for name in self.name:
            self.variance[name] = 1     

    


class DesignVariables(VariablesWithIndices):
    """
    Define design variables 
    """
    def __init__(self):
        super().__init__()

    def specify(self, self_define_res):

        self.name = super().specify(self_define_res)

    def add_variables(self, var_name, indices=None, time_index_position=None, values=None, 
                     lower_bounds = None, upper_bounds = None):
        """

        Parameters
        -----------
        var_name: a ``list`` of var names 
        indices: a ``dict`` containing indexes 
            if default (None), no extra indexes needed for all var in var_name
            for e.g., {0:["CA", "CB", "CC"], 1: [1,2,3]}. 
        time_index_position: an integer indicates which index is the time index  
            for e.g., 1 is the time index position in the indices example. 
        values: a ``list`` containing values which has the same shape of flattened variables 
            default choice is None, means there is no give nvalues 
        lower_bounds: a ``list `` of lower bounds. If given a scalar number, it is set as the lower bounds for all variables.
        upper_bounds: a ``list`` of upper bounds. If given a scalar number, it is set as the upper bounds for all variables.
        """
        self.name =  super().add_variables(var_name=var_name, indices=indices, time_index_position=time_index_position, 
                                           values=values,  lower_bounds = lower_bounds, upper_bounds = upper_bounds)
                            

    def update_values(self, values):
        return super().update_values(values)
