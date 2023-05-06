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
        self.variable_names_value = {}
        self.lower_bounds = {}
        self.upper_bounds = {}

    def set_variable_name_list(self, self_define_res):
        """
        Specify variable names with its full name.

        Parameter
        ---------
        self_define_res: a ``list`` of ``string``, containing the variable names with indexes,
            for e.g. "C['CA', 23, 0]".
        """
        self.variable_names.extend(self_define_res)

    def add_variables(
        self,
        var_name,
        indices=None,
        time_index_position=None,
        values=None,
        lower_bounds=None,
        upper_bounds=None,
    ):
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
        added_names = self._generate_variable_names_with_indices(
            var_name, indices=indices, time_index_position=time_index_position
        )

        self._check_valid_input(
            len(added_names),
            var_name,
            indices,
            time_index_position,
            values,
            lower_bounds,
            upper_bounds,
        )

        if values:
            # this dictionary keys are special set, values are its value
            self.variable_names_value.update(zip(added_names, values))

        if lower_bounds:
            if type(lower_bounds) in [int, float]:
                lower_bounds = [lower_bounds] * len(added_names)
            self.lower_bounds.update(zip(added_names, lower_bounds))

        if upper_bounds:
            if type(upper_bounds) in [int, float]:
                upper_bounds = [upper_bounds] * len(added_names)
            self.upper_bounds.update(zip(added_names, upper_bounds))

        return added_names

    def _generate_variable_names_with_indices(self, var_name, indices=None, time_index_position=None):
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
        all_index_list = []  # contains all index lists
        if indices:
            for index_pointer in indices:
                all_index_list.append(indices[index_pointer])

        # all index list for one variable, such as ["CA", 10, 1]
        # exhaustively enumerate over the full product of indices. For e.g.,
        # {0:["CA", "CB", "CC"], 1: [1,2,3]}
        # becomes ["CA", 1], ["CA", 2], ..., ["CC", 2], ["CC", 3]
        all_variable_indices = list(itertools.product(*all_index_list))

        # list store all names added this time
        added_names = []
        # iterate over index combinations ["CA", 1], ["CA", 2], ..., ["CC", 2], ["CC", 3]
        for index_instance in all_variable_indices:
            var_name_index_string = var_name + "["
            for i, idx in enumerate(index_instance):
                var_name_index_string += str(idx)

                # if i is the last index, close the []. if not, add a "," for the next index.
                if i == len(index_instance) - 1:
                    var_name_index_string += "]"
                else:
                    var_name_index_string += ","

            self.variable_names.append(var_name_index_string)
            added_names.append(var_name_index_string)

        return added_names

    def _check_valid_input(
        self,
        len_indices,
        var_name,
        indices,
        time_index_position,
        values,
        lower_bounds,
        upper_bounds,
    ):
        """
        Check if the measurement information provided are valid to use.
        """
        assert type(var_name) is str, "var_name should be a string."

        if time_index_position not in indices:
            raise ValueError("time index cannot be found in indices.")

        # if given a list, check if bounds have the same length with flattened variable
        if values and len(values) != len_indices:
            raise ValueError("Values is of different length with indices.")

        if (
            lower_bounds
            and type(lower_bounds) == list
            and len(lower_bounds) != len_indices
        ):
            raise ValueError("Lowerbounds is of different length with indices.")

        if (
            upper_bounds
            and type(upper_bounds) == list
            and len(upper_bounds) != len_indices
        ):
            raise ValueError("Upperbounds is of different length with indices.")


class MeasurementVariables(VariablesWithIndices):
    def __init__(self):
        """
        This class stores information on which algebraic and differential variables in the Pyomo model are considered measurements.
        """
        super().__init__()
        self.variance = {}

    def set_variable_name_list(self, self_define_res, variance=1):
        """
        Specify variable names with its full name.

        Parameter
        ---------
        self_define_res: a ``list`` of ``string``, containing the variable names with indexes,
            for e.g. "C['CA', 23, 0]".
        variance: a ``list`` of scalar numbers , which is the variance for this measurement.
        """
        super().set_variable_name_list(self_define_res)
        self.name = self.variable_names

        # add variance
        if variance is not list:
            variance = [variance] * len(self_define_res)

        self.variance.update(zip(self_define_res, variance))

    def add_variables(
        self, var_name, indices=None, time_index_position=None, variance=1
    ):
        """
        Parameters
        -----------
        var_name: a ``list`` of var names
        indices: a ``dict`` containing indexes
            if default (None), no extra indexes needed for all var in var_name
            for e.g., {0:["CA", "CB", "CC"], 1: [1,2,3]}.
        time_index_position: an integer indicates which index is the time index
            for e.g., 1 is the time index position in the indices example.
        variance: a scalar number, which is the variance for this measurement.
        """
        added_names = super().add_variables(
            var_name=var_name, indices=indices, time_index_position=time_index_position
        )

        self.name = self.variable_names

        # store variance
        if variance is not list:
            variance = [variance] * len(added_names)
        self.variance.update(zip(added_names, variance))

    def check_subset(self, subset_object):
        """
        Check if subset_object is a subset of the current measurement object

        Parameters
        ----------
        subset_object: a measurement object
        """
        for name in subset_object.name:
            if name not in self.name:
                raise ValueError("Measurement not in the set: ", name)

        return True


class DesignVariables(VariablesWithIndices):
    """
    Define design variables
    """

    def __init__(self):
        super().__init__()

    def set_variable_name_list(self, self_define_res):
        """
        Specify variable names with its full name.

        Parameter
        ---------
        self_define_res: a ``list`` of ``string``, containing the variable names with indexes,
            for e.g. "C['CA', 23, 0]".
        """
        super().set_variable_name_list(self_define_res)
        self.name = self.variable_names

    def add_variables(
        self,
        var_name,
        indices=None,
        time_index_position=None,
        values=None,
        lower_bounds=None,
        upper_bounds=None,
    ):
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
        super().add_variables(
            var_name=var_name,
            indices=indices,
            time_index_position=time_index_position,
            values=values,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

        self.name = self.variable_names

    def update_values(self, values):
        """
        Update values of variables. Used for defining values for design variables of different experiments.

        Parameters
        ---------
         values: a ``list`` containing values which has the same shape of flattened variables
            default choice is None, means there is no give nvalues
        """
        self.variable_names_value.update(zip(self.variable_names, values))
