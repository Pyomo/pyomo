#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import scipy_available
from pyomo.contrib.matching.maximum_matching import maximum_matching
from pyomo.contrib.matching.block_triangularize import block_triangularize
if scipy_available:
    from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
    import scipy as sp


def _check_unindexed(complist):
    for comp in complist:
        if comp.is_indexed():
            raise ValueError(
                    "Variables and constraints must be unindexed "
                    "ComponentData objects. Got %s, which is indexed."
                    % comp.name
                    )


def get_structural_incidence_matrix(variables, constraints, include_fixed=True):
    """
    This function gets the incidence matrix of Pyomo constraints and variables.

    Arguments
    ---------
    variables: A list of Pyomo variable data objects
    constraints: A list of Pyomo constraint data objects

    Returns
    -------
    A scipy.sparse coo matrix. Rows are indices into the user-provided list of
    constraints, columns are indices into the user-provided list of variables.
    Entries are 1.0.

    """
    _check_unindexed(variables+constraints)
    N, M = len(variables), len(constraints)
    var_idx_map = ComponentMap((v, i) for i, v in enumerate(variables))
    rows = []
    cols = []
    for i, con in enumerate(constraints):
        cols.extend(var_idx_map[v] for v in
                identify_variables(con.body, include_fixed=include_fixed)
                if v in var_idx_map)
        rows.extend([i]*(len(cols) - len(rows)))
    assert len(rows) == len(cols)
    data = [1.0]*len(rows)
    matrix = sp.sparse.coo_matrix( (data, (rows, cols)), shape=(M, N) )
    return matrix


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

        _check_unindexed(variables+constraints)
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
        # Switch the order of the maps here to match the method call.
        # Hopefully this does not get too confusing...
        return var_block_map, con_block_map
