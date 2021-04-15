#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.collections import ComponentMap
from pyomo.contrib.matching.maximum_matching import maximum_matching
from pyomo.contrib.matching.block_triangularize import block_triangularize
from pyomo.common.dependencies import scipy_available
if scipy_available:
    from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP

class IncidenceGraphInterface(object):
    """
    The purpose of this class is to allow the user to easily
    analyze graphs of variables and contraints in a Pyomo
    model without constructing multiple PyomoNLPs.
    """

    def __init__(self, model):
        self.nlp = PyomoNLP(model)

    def _validate_input(self, variables, constraints):
        if variables is None:
            variables = self.nlp.get_pyomo_variables()
        if constraints is None:
            constraints = self.nlp.get_pyomo_constraints()

        for comp in variables+constraints:
            if comp.is_indexed():
                raise ValueError(
                        "Variables and constraints must be unindexed "
                        "ComponentData objects. Got %s, which is indexed."
                        % comp.name
                        )
        return variables, constraints

    def maximum_matching(self, variables=None, constraints=None):
        """
        Returns a maximal matching between the constraints and variables,
        in terms of a map from constraints to variables.
        """
        variables, constraints = self._validate_input(variables, constraints)

        matrix = self.nlp.extract_submatrix_jacobian(variables, constraints)
        matching = maximum_matching(matrix.tocoo())
        # Matching maps row (constraint) indices to column (variable) indices

        return ComponentMap((constraints[i], variables[j])
                for i, j in matching.items())

    def block_triangularize(self, variables=None, constraints=None):
        """
        Returns two ComponentMaps. A map from variables to their blocks
        in a block triangularization of the incidence matrix, and a
        map from constraints to their blocks in a block triangularization
        of the incidence matrix.
        """
        variables, constraints = self._validate_input(variables, constraints)

        matrix = self.nlp.extract_submatrix_jacobian(variables, constraints)
        row_block_map, col_block_map = block_triangularize(matrix.tocoo())
        con_block_map = ComponentMap((constraints[i], idx)
                for i, idx in row_block_map.items())
        var_block_map = ComponentMap((variables[j], idx)
                for j, idx in col_block_map.items())
        # Switch the order of the maps here to matching the method call.
        # Hopefully this does not get too confusing...
        return var_block_map, con_block_map
