from pyomo.environ import (Var,
                           SOSConstraint,
                           Constraint,
                           Objective,
                           value,
                           SortComponents,
                           ComponentMap)
from pyomo.repn.canonical_repn import \
    generate_canonical_repn

def to_matrix_form(model):
    """
    Converts a concrete Pyomo model with a linear objective
    and linear constraints into matrix form.

    Args:
        model: A concrete Pyomo model.

    Returns:
        Objects that define the following LP representation:

        min(max)    c0 + c^T x
        s.t.     bL <= Ax <= bU
                 xL <=  x <= xU

        where,

          c0: scalar representing the aggregation of all
              constants found in the objective expression
          c:  nvars-length list of objective coefficients.
          bL: ncons-length list of constraint upper bounds
          bU: ncons-length list of constraint lower bounds
          A:  3-tuple consisting of list objects (data,
              indices, indptr) defining a sparse matrix in
              Compressed Sparse Row format
          xL: nvars-length list of variable lower bounds
          xU: nvars-length list of variable upper bounds

        In addition, the following mapping objects are
        returned:

          vartocol: maps model variable objects to their
                    integer column index in the A     
                    matrix. E.g.,
                       vartocol[model.x[5]] # -> 0
          contorow: maps model constraint objects to their
                    integer row index in the A matrix. E.g.,
                       contorow[model.c] # -> 19

        All variable and constraint bound vectors will
        contain values of float('-inf') and float('inf')
        where the corresponding bound does not exist.
    """

    # Assign each variable a deterministic symbol (an index
    # in a list) so that we can guarantee the same matrix
    # ordering for a given Pyomo model. We can not assign a
    # column index until after collecting the list of
    # variables that are actually used.
    sortOrder = (SortComponents.indices |
                 SortComponents.alphabetical)
    all_blocks = [_b for _b in
                  model.block_data_objects(
                      active=True,
                      sort=sortOrder)]
    VarSymbolToVarObject = []
    for block in all_blocks:
        VarSymbolToVarObject.extend(
            block.component_data_objects(
                Var,
                sort=sortOrder,
                descend_into=False))
    VarIDToVarSymbol = \
        dict((id(var), index) for index, var in
             enumerate(VarSymbolToVarObject))

    # Loop over objective and constraints to generate the
    # cost vector and matrix rows. Raise an exception if any
    # nonlinear expressions are encountered.
    negative_infinity = float('-inf')
    positive_infinity = float('inf')
    nobjs = 0
    referenced_var_symbols = set()
    A_indptr = [0]
    A_indices = []
    A_data = []
    bL = []
    bU = []
    c_sparse = {}
    c0 = 0.0
    RowIndexToConstraintObject = []
    for block in all_blocks:

        for objective in block.component_data_objects(
                Objective,
                active=True,
                sort=sortOrder,
                descend_into=False):

            nobjs += 1
            if nobjs > 1:
                raise ValueError(
                    "This function does not support "
                    "multiple objectives")

            polynomial_degree = \
                objective.expr.polynomial_degree()
            if (polynomial_degree != 0) and \
               (polynomial_degree != 1):
                raise ValueError(
                    "This function does not support "
                    "nonlinear objectives")

            canonical_repn = \
                generate_canonical_repn(objective.expr)
            variables = canonical_repn.variables
            coefficients = canonical_repn.linear
            if variables is not None:
                for var, coef in zip(variables,
                                     coefficients):
                    var_symbol = VarIDToVarSymbol[id(var)]
                    c_sparse[var_symbol] = coef
                    referenced_var_symbols.add(var_symbol)

            if canonical_repn.constant is not None:
                c0 = value(canonical_repn.constant)

        for sosconstraint in block.component_data_objects(
                SOSConstraint,
                active=True,
                sort=sortOrder,
                descend_into=False):
            raise ValueError("This function does not "
                             "support SOSConstraints")

        for constraint in block.component_data_objects(
                Constraint,
                active=True,
                sort=sortOrder,
                descend_into=False):

            polynomial_degree = \
                constraint.body.polynomial_degree()
            if (polynomial_degree != 0) and \
               (polynomial_degree != 1):
                raise ValueError(
                    "This function does not support "
                    "nonlinear constraints")

            RowIndexToConstraintObject.append(constraint)

            canonical_repn = \
                generate_canonical_repn(constraint.body)
            variables = canonical_repn.variables
            coefficients = canonical_repn.linear

            row_variable_symbols = []
            row_coefficients = []
            if variables is not None:
                row_variable_symbols = \
                    [VarIDToVarSymbol[id(var)]
                     for var in variables]
                referenced_var_symbols.\
                    update(row_variable_symbols)
                row_coefficients = coefficients

            A_indptr.append(A_indptr[-1] +
                            len(row_variable_symbols))
            A_indices.extend(row_variable_symbols)
            A_data.extend(row_coefficients)

            L = negative_infinity
            U = positive_infinity
            constant = 0.0

            if constraint.lower is not None:
                L = value(constraint.lower)
            if constraint.upper is not None:
                U = value(constraint.upper)
            if canonical_repn.constant is not None:
                constant = value(canonical_repn.constant)

            bL.append(L - constant)
            bU.append(U - constant)

    ncols = len(referenced_var_symbols)

    # Assign a column index to the set of referenced
    # variables
    ColumnIndexToVarSymbol = sorted(referenced_var_symbols)
    VarSymbolToColumnIndex = \
        dict((symbol, col) for col, symbol in
             enumerate(ColumnIndexToVarSymbol))

    A_indices = [VarSymbolToColumnIndex[symbol]
                 for symbol in A_indices]

    ColumnIndexToVarObject = \
        [VarSymbolToVarObject[var_symbol]
         for var_symbol in ColumnIndexToVarSymbol]

    # Convert the sparse cost vector into a dense list based
    # on the variable column id assignments.
    c = [0.0 for j in range(ncols)]
    for var_symbol, coef in c_sparse.items():
        c[VarSymbolToColumnIndex[var_symbol]] = coef

    # Generate dense xL and xU variable bound lists based on
    # the variable column id assignments
    xL = [negative_infinity for j in range(ncols)]
    xU = [positive_infinity for j in range(ncols)]
    for j, var in enumerate(ColumnIndexToVarObject):
        if var.lb is not None:
            xL[j] = value(var.lb)
        if var.ub is not None:
            xU[j] = value(var.ub)

    # Generate the component maps that allow one to recover
    # the row/column index from a constraint/variable
    # object. The reverse maps are easy enough to generate
    # from these two maps if needed.
    vartocol = ComponentMap(
        (var, j) for j, var in
        enumerate(ColumnIndexToVarObject))
    contorow = ComponentMap(
        (con, i)
        for i, con in enumerate(RowIndexToConstraintObject))
    return (c0, c,
            bL, bU,
            (A_data, A_indices, A_indptr),
            xL, xU,
            vartocol, contorow)
