#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import enum
from pyomo.core.base.block import Block
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.objective import Objective
from pyomo.core.base.reference import Reference
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.dependencies import scipy_available
from pyomo.common.dependencies import networkx as nx
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.triangularize import block_triangularize
from pyomo.contrib.incidence_analysis.dulmage_mendelsohn import (
    dulmage_mendelsohn,
    )
if scipy_available:
    from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
    import scipy as sp


class IncidenceMatrixType(enum.Enum):
    NONE = 0
    STRUCTURAL = 1
    NUMERIC = 2


def _check_unindexed(complist):
    for comp in complist:
        if comp.is_indexed():
            raise ValueError(
                    "Variables and constraints must be unindexed "
                    "ComponentData objects. Got %s, which is indexed."
                    % comp.name
                    )


def get_incidence_graph(variables, constraints, include_fixed=True):
    """
    This function gets the incidence graph of Pyomo variables and constraints.

    Arguments:
    ----------
    variables: List of Pyomo VarData objects
        Variables that will appear in incidence graph
    constraints: List of Pyomo ConstraintData objects
        Constraints that will appear in incidence graph
    include_fixed: Bool
        Flag for whether fixed variable should be included in the incidence

    Returns:
    --------
    NetworkX Graph
        
    """
    _check_unindexed(variables+constraints)
    N, M = len(variables), len(constraints)
    graph = nx.Graph()
    graph.add_nodes_from(range(M), bipartite=0)
    graph.add_nodes_from(range(M, M+N), bipartite=1)
    var_node_map = ComponentMap((v, M+i) for i, v in enumerate(variables))
    for i, con in enumerate(constraints):
        for var in identify_variables(con.body, include_fixed=include_fixed):
            if var in var_node_map:
                graph.add_edge(i, var_node_map[var])
    return graph


def get_structural_incidence_matrix(variables, constraints, include_fixed=True):
    """
    This function gets the incidence matrix of Pyomo constraints and variables.

    Arguments
    ---------
    variables: List of Pyomo VarData objects
    constraints: List of Pyomo ConstraintData objects
    include_fixed: Bool
        Flag for whether fixed variables should be included in the matrix
        nonzeros

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


def get_numeric_incidence_matrix(variables, constraints):
    """
    This function gets the numeric incidence matrix (Jacobian) of Pyomo
    constraints with respect to variables.
    """
    # NOTE: There are several ways to get a numeric incidence matrix
    # from a Pyomo model. This function implements a somewhat roundabout
    # method, which is to construct a dummy Block with the necessary
    # variables and constraints, then construct a PyNumero PyomoNLP
    # from the block and have PyNumero evaluate the desired Jacobian
    # via ASL.
    comps = list(variables) + list(constraints)
    _check_unindexed(comps)
    M, N = len(constraints), len(variables)
    _block = Block()
    _block.construct()
    _block.obj = Objective(expr=0)
    _block.vars = Reference(variables)
    _block.cons = Reference(constraints)
    var_set = ComponentSet(variables)
    other_vars = []
    for con in constraints:
        for var in identify_variables(con.body, include_fixed=False):
            # Fixed vars will be ignored by the nl file write, so
            # there is no point to including them here.
            # A different method of assembling this matrix, e.g.
            # Pyomo's automatic differentiation, could support taking
            # derivatives with respect to fixed variables.
            if var not in var_set:
                other_vars.append(var)
                var_set.add(var)
    # These variables are necessary due to the nl writer's philosophy
    # about what constitutes a model. Note that we take derivatives with
    # respect to them even though this is not necessary. We could fix them
    # here to avoid doing this extra work, but that would alter the user's
    # model, which we would rather not do.
    _block.other_vars = Reference(other_vars)
    _nlp = PyomoNLP(_block)
    return _nlp.extract_submatrix_jacobian(variables, constraints)


class IncidenceGraphInterface(object):
    """
    The purpose of this class is to allow the user to easily
    analyze graphs of variables and contraints in a Pyomo
    model without constructing multiple PyomoNLPs.
    """

    def __init__(self, model=None):
        """
        """
        # If the user gives us a model or an NLP, we assume they want us
        # to cache the incidence matrix for fast analysis of submatrices
        # later on.
        # WARNING: This cache will become invalid if the user alters their
        #          model.
        if model is None:
            self.cached = IncidenceMatrixType.NONE
        elif isinstance(model, PyomoNLP):
            nlp = model
            self.cached = IncidenceMatrixType.NUMERIC
            self.variables = nlp.get_pyomo_variables()
            self.constraints = nlp.get_pyomo_constraints()
            self.var_index_map = ComponentMap(
                    (var, idx) for idx, var in enumerate(self.variables))
            self.con_index_map = ComponentMap(
                    (con, idx) for idx, con in enumerate(self.constraints))
            self.incidence_matrix = nlp.evaluate_jacobian_eq()
        elif isinstance(model, Block):
            self.cached = IncidenceMatrixType.STRUCTURAL
            self.variables = list(model.component_data_objects(Var))
            self.constraints = list(model.component_data_objects(Constraint))
            self.var_index_map = ComponentMap(
                    (var, i) for i, var in enumerate(self.variables))
            self.con_index_map = ComponentMap(
                    (con, i) for i, con in enumerate(self.constraints))
            self.incidence_matrix = get_structural_incidence_matrix(
                    self.variables,
                    self.constraints,
                    )
        else:
            raise TypeError(
                "Unsupported type for incidence matrix. Expected "
                "%s or %s but got %s."
                % (PyomoNLP, Block, type(model))
                )

    def _validate_input(self, variables, constraints):
        if variables is None:
            if self.cached is IncidenceMatrixType.NONE:
                raise ValueError(
                        "Neither variables nor a model have been provided."
                        )
            else:
                variables = self.variables
        if constraints is None:
            if self.cached is IncidenceMatrixType.NONE:
                raise ValueError(
                        "Neither constraints nor a model have been provided."
                        )
            else:
                constraints = self.constraints

        _check_unindexed(variables+constraints)
        return variables, constraints

    def _extract_submatrix(self, variables, constraints):
        # Assumes variables and constraints are valid
        if self.cached is IncidenceMatrixType.NONE:
            return get_structural_incidence_matrix(
                    variables,
                    constraints,
                    include_fixed=False,
                    )
        else:
            N, M = len(variables), len(constraints)
            old_new_var_indices = dict((self.var_index_map[v], i)
                    for i, v in enumerate(variables))
            old_new_con_indices = dict((self.con_index_map[c], i)
                    for i, c in enumerate(constraints))
            coo = self.incidence_matrix
            new_row = []
            new_col = []
            new_data = []
            for r, c, e in zip(coo.row, coo.col, coo.data):
                if r in old_new_con_indices and c in old_new_var_indices:
                    new_row.append(old_new_con_indices[r])
                    new_col.append(old_new_var_indices[c])
                    new_data.append(e)
            return sp.sparse.coo_matrix(
                    (new_data, (new_row, new_col)),
                    shape=(M, N),
                    )

    def maximum_matching(self, variables=None, constraints=None):
        """
        Returns a maximal matching between the constraints and variables,
        in terms of a map from constraints to variables.
        """
        variables, constraints = self._validate_input(variables, constraints)
        matrix = self._extract_submatrix(variables, constraints)

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
        matrix = self._extract_submatrix(variables, constraints)

        row_block_map, col_block_map = block_triangularize(matrix.tocoo())
        con_block_map = ComponentMap((constraints[i], idx)
                for i, idx in row_block_map.items())
        var_block_map = ComponentMap((variables[j], idx)
                for j, idx in col_block_map.items())
        # Switch the order of the maps here to match the method call.
        # Hopefully this does not get too confusing...
        return var_block_map, con_block_map

    def dulmage_mendelsohn(self, variables=None, constraints=None):
        """
        Returns the Dulmage-Mendelsohn partition of the incidence graph
        of the provided variables and constraints.

        Returns:
        --------
        ColPartition namedtuple and RowPartition namedtuple.
        The ColPartition is returned first to match the order of variables
        and constraints in the method arguments.
        These partition variables (columns) and constraints (rows)
        into overconstrained, underconstrained, unmatched, and square.

        """
        variables, constraints = self._validate_input(variables, constraints)
        matrix = self._extract_submatrix(variables, constraints)

        row_partition, col_partition = dulmage_mendelsohn(matrix.tocoo())
        con_partition = tuple(
                [constraints[i] for i in subset] for subset in row_partition
                )
        var_partition = tuple(
                [variables[i] for i in subset] for subset in col_partition
                )
        # Switch the order of the maps here to match the method call.
        # Hopefully this does not get too confusing...
        return var_partition, con_partition
