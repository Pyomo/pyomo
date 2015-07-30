#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

#
# NOTE: deprecated code
#
# This code is here for historical purposes, and
# because we may support an explicit matrix representation for models.
#

from pyomo.core.base import Objective, Constraint
import array

from six.moves import xrange

def to_standard_form(self):
    """
    Produces a standard-form representation of the model. Returns
    the coefficient matrix (A), the cost vector (c), and the
    constraint vector (b), where the 'standard form' problem is

    min/max c'x
    s.t.    Ax = b
            x >= 0

    All three returned values are instances of the array.array
    class, and store Python floats (C doubles).
    """

    from pyomo.repn import generate_canonical_repn


    # We first need to create an map of all variables to their column
    # number
    colID = {}
    ID2name = {}
    id = 0
    tmp = self.variables().keys()
    tmp.sort()

    for v in tmp:
        colID[v] = id
        ID2name[id] = v
        id += 1

    # First we go through the constraints and introduce slack and excess
    # variables to eliminate inequality constraints
    #
    # N.B. Structure heirarchy:
    #
    # active_components: {class: {attr_name: object}}
    # object -> Constraint: ._data: {ndx: _ConstraintData}
    # _ConstraintData: .lower, .body, .upper
    #
    # So, altogether, we access a lower bound via
    #
    # model.component_map(active=True)[Constraint]['con_name']['index'].lower
    #
    # {le,ge,eq}Constraints are
    # {constraint_name: {index: {variable_or_none: coefficient}} objects
    # that represent each constraint. None in the innermost dictionary
    # represents the constant term.
    #
    # i.e.
    #
    # min  x1 + 2*x2 +          x4
    # s.t. x1                         = 1
    #           x2   + 3*x3          <= -1
    #      x1 +                 x4   >= 3
    #      x1 + 2*x2 +      +   3*x4 >= 0
    #
    #
    # would be represented as (modulo the names of the variables,
    # constraints, and indices)
    #
    # eqConstraints = {'c1': {None: {'x1':1, None:-1}}}
    # leConstraints = {'c2': {None: {'x2':1, 'x3':3, None:1}}}
    # geConstraints = {'c3': {None: {'x1':1, 'x4':1, None:-3}},
    #                  'c4': {None: {'x1':1, 'x2':2, 'x4':1, None:0}}}
    #
    # Note the we have the luxury of dealing only with linear terms.
    var_id_map = {}
    leConstraints = {}
    geConstraints = {}
    eqConstraints = {}
    objectives = {}
    # For each registered component
    for c in self.component_map(active=True):

        # Get all subclasses of Constraint
        if issubclass(c, Constraint):
            cons = self.component_map(c, active=True)

            # Get the name of the constraint, and the constraint set itself
            for con_set_name in cons:
                con_set = cons[con_set_name]

                # For each indexed constraint in the constraint set
                for ndx in con_set._data:
                    con = con_set._data[ndx]

                    # Process the body
                    terms = self._process_canonical_repn(
                        generate_canonical_repn(con.body, var_id_map))

                    # Process the bounds of the constraint
                    if con.equality:
                        # Equality constraint, only check lower bound
                        lb = self._process_canonical_repn(
                            generate_canonical_repn(con.lower, var_id_map))

                        # Update terms
                        for k in lb:
                            v = lb[k]
                            if k in terms:
                                terms[k] -= v
                            else:
                                terms[k] = -v

                        # Add constraint to equality constraints
                        eqConstraints[(con_set_name, ndx)] = terms
                    else:

                        # Process upper bounds (<= constraints)
                        if con.upper is not None:
                            # Less than or equal to constraint
                            tmp = dict(terms)

                            ub = self._process_canonical_repn(
                                generate_canonical_repn(con.upper, var_id_map))

                            # Update terms
                            for k in ub:
                                if k in terms:
                                    tmp[k] -= ub[k]
                                else:
                                    tmp[k] = -ub[k]

                            # Add constraint to less than or equal to
                            # constraints
                            leConstraints[(con_set_name, ndx)] = tmp

                        # Process lower bounds (>= constraints)
                        if con.lower is not None:
                            # Less than or equal to constraint
                            tmp = dict(terms)

                            lb = self._process_canonical_repn(
                                generate_canonical_repn(con.lower, var_id_map))

                            # Update terms
                            for k in lb:
                                if k in terms:
                                    tmp[k] -= lb[k]
                                else:
                                    tmp[k] = -lb[k]

                            # Add constraint to less than or equal to
                            # constraints
                            geConstraints[(con_set_name, ndx)] = tmp
        elif issubclass(c, Objective):
            # Process objectives
            objs = self.component_map(c, active=True)

            # Get the name of the objective, and the objective set itself
            for obj_set_name in objs:
                obj_set = objs[obj_set_name]

                # For each indexed objective in the objective set
                for ndx in obj_set._data:
                    obj = obj_set._data[ndx]
                    # Process the objective
                    terms = self._process_canonical_repn(
                        generate_canonical_repn(obj.expr, var_id_map))

                    objectives[(obj_set_name, ndx)] = terms


    # We now have all the constraints. Add a slack variable for every
    # <= constraint and an excess variable for every >= constraint.
    nSlack = len(leConstraints)
    nExcess = len(geConstraints)

    nConstraints = len(leConstraints) + len(geConstraints) + \
                   len(eqConstraints)
    nVariables = len(colID) + nSlack + nExcess
    nRegVariables = len(colID)

    # Make the arrays
    coefficients = array.array("d", [0]*nConstraints*nVariables)
    constraints = array.array("d", [0]*nConstraints)
    costs = array.array("d", [0]*nVariables)

    # Populate the coefficient matrix
    constraintID = 0

    # Add less than or equal to constraints
    for ndx in leConstraints:
        con = leConstraints[ndx]
        for termKey in con:
            coef = con[termKey]

            if termKey is None:
                # Constraint coefficient
                constraints[constraintID] = -coef
            else:
                # Variable coefficient
                col = colID[termKey]
                coefficients[constraintID*nVariables + col] = coef

        # Add the slack
        coefficients[constraintID*nVariables + nRegVariables + \
                    constraintID] = 1
        constraintID += 1

    # Add greater than or equal to constraints
    for ndx in geConstraints:
        con = geConstraints[ndx]
        for termKey in con:
            coef = con[termKey]

            if termKey is None:
                # Constraint coefficient
                constraints[constraintID] = -coef
            else:
                # Variable coefficient
                col = colID[termKey]
                coefficients[constraintID*nVariables + col] = coef

        # Add the slack
        coefficients[constraintID*nVariables + nRegVariables + \
                    constraintID] = -1
        constraintID += 1

    # Add equality constraints
    for ndx in eqConstraints:
        con = eqConstraints[ndx]
        for termKey in con:
            coef = con[termKey]

            if termKey is None:
                # Constraint coefficient
                constraints[constraintID] = -coef
            else:
                # Variable coefficient
                col = colID[termKey]
                coefficients[constraintID*nVariables + col] = coef

        constraintID += 1

    # Determine cost coefficients
    for obj_name in objectives:
        obj = objectives[obj_name]()
        for var in obj:
            costs[colID[var]] = obj[var]

    # Print the model
    #
    # The goal is to print
    #
    #         var1   var2   var3   ...
    #       +--                     --+
    #       | cost1  cost2  cost3  ...|
    #       +--                     --+
    #       +--                     --+ +-- --+
    # con1  | coef11 coef12 coef13 ...| | eq1 |
    # con2  | coef21 coef22 coef23 ...| | eq2 |
    # con2  | coef31 coef32 coef33 ...| | eq3 |
    #  .    |   .      .      .   .   | |  .  |
    #  .    |   .      .      .    .  | |  .  |
    #  .    |   .      .      .     . | |  .  |

    constraintPadding = 2
    numFmt = "% 1.4f"
    altFmt = "% 1.1g"
    maxColWidth = max(len(numFmt % 0.0), len(altFmt % 0.0))
    maxConstraintColWidth = max(len(numFmt % 0.0), len(altFmt % 0.0))

    # Generate constraint names
    maxConNameLen = 0
    conNames = []
    for name in leConstraints:
        strName = str(name)
        if len(strName) > maxConNameLen:
            maxConNameLen = len(strName)
        conNames.append(strName)
    for name in geConstraints:
        strName = str(name)
        if len(strName) > maxConNameLen:
            maxConNameLen = len(strName)
        conNames.append(strName)
    for name in eqConstraints:
        strName = str(name)
        if len(strName) > maxConNameLen:
            maxConNameLen = len(strName)
        conNames.append(strName)

    # Generate the variable names
    varNames = [None]*len(colID)
    for name in colID:
        tmp_name = " " + name
        if len(tmp_name) > maxColWidth:
            maxColWidth = len(tmp_name)
        varNames[colID[name]] = tmp_name
    for i in xrange(0, nSlack):
        tmp_name = " _slack_%i" % i
        if len(tmp_name) > maxColWidth:
            maxColWidth = len(tmp_name)
        varNames.append(tmp_name)
    for i in xrange(0, nExcess):
        tmp_name = " _excess_%i" % i
        if len(tmp_name) > maxColWidth:
            maxColWidth = len(tmp_name)
        varNames.append(tmp_name)

    # Variable names
    line = " "*maxConNameLen + (" "*constraintPadding) + " "
    for col in xrange(0, nVariables):
        # Format entry
        token = varNames[col]

        # Pad with trailing whitespace
        token += " "*(maxColWidth - len(token))

        # Add to line
        line += " " + token + " "
    print(line+'\n')

    # Cost vector
    print(" "*maxConNameLen + (" "*constraintPadding) + "+--" + \
          " "*((maxColWidth+2)*nVariables - 4) + "--+" + '\n')
    line = " "*maxConNameLen + (" "*constraintPadding) + "|"
    for col in xrange(0, nVariables):
        # Format entry
        token = numFmt % costs[col]
        if len(token) > maxColWidth:
            token = altFmt % costs[col]

        # Pad with trailing whitespace
        token += " "*(maxColWidth - len(token))

        # Add to line
        line += " " + token + " "
    line += "|"
    print(line+'\n')
    print(" "*maxConNameLen + (" "*constraintPadding) + "+--" + \
          " "*((maxColWidth+2)*nVariables - 4) + "--+"+'\n')

    # Constraints
    print(" "*maxConNameLen + (" "*constraintPadding) + "+--" + \
          " "*((maxColWidth+2)*nVariables - 4) + "--+" + \
          (" "*constraintPadding) + "+--" + \
          (" "*(maxConstraintColWidth-1)) + "--+"+'\n')
    for row in xrange(0, nConstraints):
        # Print constraint name
        line = conNames[row] + (" "*constraintPadding) + (" "*(maxConNameLen - len(conNames[row]))) + "|"

        # Print each coefficient
        for col in xrange(0, nVariables):
            # Format entry
            token = numFmt % coefficients[nVariables*row + col]
            if len(token) > maxColWidth:
                token = altFmt % coefficients[nVariables*row + col]

            # Pad with trailing whitespace
            token += " "*(maxColWidth - len(token))

            # Add to line
            line += " " + token + " "

        line += "|" + (" "*constraintPadding) + "|"

        # Add constraint vector
        token = numFmt % constraints[row]
        if len(token) > maxConstraintColWidth:
            token = altFmt % constraints[row]

        # Pad with trailing whitespace
        token += " "*(maxConstraintColWidth - len(token))

        line += " " + token + "  |"
        print(line+'\n')
    print(" "*maxConNameLen + (" "*constraintPadding) + "+--" + \
          " "*((maxColWidth+2)*nVariables - 4) + "--+" + \
          (" "*constraintPadding) + "+--" + (" "*(maxConstraintColWidth-1))\
          + "--+"+'\n')

    return (coefficients, costs, constraints)

def _process_canonical_repn(self, expr):
    """
    Returns a dictionary of {var_name_or_None: coef} values
    """

    terms = {}

    # Get the variables from the canonical representation
    vars = expr.pop(-1, {})

    # Find the linear terms
    linear = expr.pop(1, {})
    for k in linear:
        # FrozeDicts don't support (k, v)-style iteration
        v = linear[k]

        # There's exactly 1 variable in each term
        terms[vars[k.keys()[0]].label] = v

    # Get the constant term, if present
    const = expr.pop(0, {})
    if None in const:
        terms[None] = const[None]

    if len(expr) != 0:
        raise TypeError("Nonlinear terms in expression")

    return terms

