#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________


import logging
import os
import glob
import sys
import re
import time
import math

import pyutilib.services
from pyutilib.misc import Bunch, Options
from pyutilib.math import infinity

import pyomo.util.plugin
from pyomo.opt.base import *
from pyomo.opt.base.solvers import _extract_version
from pyomo.opt.results import *
from pyomo.opt.solver import *
from pyomo.core.base import (SymbolMap,
                             ComponentMap,
                             NumericLabeler,
                             TextLabeler,
                             is_fixed,
                             value)
from pyomo.repn import generate_canonical_repn
from pyomo.solvers import wrappers

from six import itervalues, iterkeys, iteritems, advance_iterator
from six.moves import xrange

logger = logging.getLogger('pyomo.solvers')

try:
    unicode
except:
    basestring = str

_cplex_version = None
try:
    import cplex
    from cplex.exceptions import CplexError, CplexSolverError
    # create a version tuple of length 4
    _cplex_version = tuple(int(i) for i in cplex.Cplex().get_version().split('.'))
    while(len(_cplex_version) < 4):
        _cplex_version += (0,)
    _cplex_version = _cplex_version[:4]
    cplex_import_available=True
except ImportError:
    cplex_import_available=False
except Exception as e:
    # other forms of exceptions can be thrown by CPLEX python
    # import.  For example, an error in code invoked by the module's
    # __init__.  We should continue gracefully and not cause a fatal
    # error in Pyomo.
    print("Import of cplex failed - cplex message=%s\n" % (e,) )
    cplex_import_available=False

class CplexSolverWrapper(wrappers.MIPSolverWrapper):

    def __init__(self, solver):
        self.cplex = solver

    def add(self, constraint):
        """TODO"""
        pass


class ModelSOS(object):
    def __init__(self):
        self.sosType = {}
        self.sosName = {}
        self.varnames = {}
        self.varids = {}
        self.weights = {}
        self.block_cntr = 0

    def count_constraint(self,symbol_map,labeler,variable_symbol_map,soscondata):

        sos_items = list(soscondata.get_items())
        level = soscondata.level

        if len(sos_items) == 0:
            return

        self.block_cntr += 1
        varnames = self.varnames[self.block_cntr] = []
        varids = self.varids[self.block_cntr] = []
        weights = self.weights[self.block_cntr] = []
        if level == 1:
            self.sosType[self.block_cntr] = cplex.Cplex.SOS.type.SOS1
        elif level == 2:
            self.sosType[self.block_cntr] = cplex.Cplex.SOS.type.SOS2
        else:
            raise ValueError("Unsupported SOSConstraint level %s" % level)

        self.sosName[self.block_cntr] = symbol_map.getSymbol(soscondata,labeler)

        for vardata, weight in sos_items:
            if vardata.fixed:
                raise RuntimeError("SOSConstraint '%s' includes a fixed variable '%s'. "
                                   "This is currently not supported. Deactivate this constraint "
                                   "in order to proceed" % (soscondata.name, vardata.name))
            varids.append(id(vardata))
            varnames.append(variable_symbol_map.getSymbol(vardata))
            weights.append(weight)


class CPLEXDirect(OptSolver):
    """The CPLEX LP/MIP solver
    """

    pyomo.util.plugin.alias('_cplex_direct',
                            doc='Direct Python interface to the CPLEX LP/MIP solver')

    def __init__(self, **kwds):
        #
        # Call base class constructor
        #

        # This gets overridden by CPLEXPersistent
        if 'type' not in kwds:
            kwds['type'] = 'cplexdirect'
        OptSolver.__init__(self, **kwds)

        # this interface doesn't use files, but we can create a log
        # file is requested
        self._keepfiles = False
        # do we warmstart
        self._warm_start_solve = False
        # io_options
        self._symbolic_solver_labels = False
        self._output_fixed_variable_bounds = False
        self._skip_trivial_constraints = False

        # The working problem instance, via CPLEX python constructs.
        self._active_cplex_instance = None

        # Note: Undefined capabilities default to 'None'
        self._capabilities = Options()
        self._capabilities.linear = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = True
        self._capabilities.integer = True
        self._capabilities.sos1 = True
        self._capabilities.sos2 = True

        # flag allowing for the use, during solves, of user-defined callbacks.
        self._allow_callbacks = True

        # the CPLEX python API doesn't provide a mechanism to track
        # user/system/wall clock time, so it's up to us. stored as an
        # attribute of the plugin to facilitate persistance across
        # various portions of the method invocations.
        self._solve_user_time = None

        # collection of id(_VarData).
        self._referenced_variable_ids = set()

    def available(self, exception_flag=True):
        """ True if the solver is available """

        if exception_flag is False:
            return cplex_import_available
        else:
            if cplex_import_available is False:
                raise ApplicationError(
                    "No CPLEX <-> Python bindings available - CPLEX direct "
                    "solver functionality is not available")
            else:
                return True

    def _get_version(self):
        if _cplex_version is None:
            return _extract_version('')
        return _cplex_version

    def _get_bound(self, exp):
        if exp is None:
            return None
        if is_fixed(exp):
            return value(exp)
        raise ValueError("non-fixed bound: " + str(exp))

    #
    # CPLEX requires objective expressions to be specified via
    # something other than a sparse pair!
    # NOTE: The returned offset is guaranteed to be a float.
    #
    def _encode_constraint_body_linear(self, expression, labeler, as_pairs=False):

        variables = [] # string names of variables
        coefficients = [] # variable coefficients

        pairs = []

        hash_to_variable_map = expression[-1]
        self_variable_symbol_map = self._variable_symbol_map

        if 1 in expression:
            for var_hash, var_coefficient in iteritems(expression[1]):

                vardata = hash_to_variable_map[var_hash]
                self._referenced_variable_ids.add(id(vardata))
                variable_name = self_variable_symbol_map.getSymbol(vardata)

                if as_pairs is True:
                    pairs.append((variable_name, var_coefficient))
                else:
                    variables.append(variable_name)
                    coefficients.append(var_coefficient)

        offset=0.0
        if 0 in expression:
            offset = expression[0][None]

        if as_pairs is True:
            return pairs, offset
        else:
            expr = cplex.SparsePair(ind=variables, val=coefficients)
            return expr, offset

    #
    # CPLEX requires objective expressions to be specified via
    # something other than a sparse pair!
    # NOTE: The returned offset is guaranteed to be a float.
    # NOTE: This function is a variant of the above, specialized
    #       for LinearCanonicalRepn objects.
    #
    def _encode_constraint_body_linear_specialized(self,
                                                   linear_repn,
                                                   labeler,
                                                   use_variable_names=True,
                                                   cplex_variable_name_index_map=None,
                                                   as_pairs=False):

        variable_identifiers = [] # strings if use_variable_names = True; integers otherwise.
        variable_coefficients = []
        pairs = []

        # caching for efficiency
        constant = linear_repn.constant
        coefficients = linear_repn.linear
        variables = linear_repn.variables

        self_variable_symbol_map = self._variable_symbol_map

        if variables != None:

            for var_value, var_coefficient in zip(variables, coefficients):

                self._referenced_variable_ids.add(id(var_value))
                variable_name = self_variable_symbol_map.getSymbol(var_value)

                if use_variable_names == False:
                    cplex_variable_id = cplex_variable_name_index_map[variable_name]

                if as_pairs == True:

                    if use_variable_names == True:
                        pairs.append((variable_name, var_coefficient))
                    else:
                        pairs.append((cplex_variable_id, var_coefficient))

                else:

                    if use_variable_names == True:
                        variable_identifiers.append(variable_name)
                    else:
                        variable_identifiers.append(cplex_variable_id)
                    variable_coefficients.append(var_coefficient)

        offset=0.0
        if constant is not None:
            offset = constant

        if as_pairs is True:
            return pairs, offset
        else:
            expr = cplex.SparsePair(ind=variable_identifiers, val=variable_coefficients)
            return expr, offset

    #
    # Handle quadratic constraints and objectives
    #
    def _encode_constraint_body_quadratic(self,
                                          expression,
                                          labeler,
                                          as_triples=False,
                                          is_obj=1.0):

        variables1 = [] # string names of variables
        variables2 = [] # string names of variables
        coefficients = [] # variable coefficients

        triples = []

        hash_to_variable_map = expression[-1]

        self_variable_symbol_map = self._variable_symbol_map
        for vrs, coeff in iteritems(expression[2]):

            variable_hash_iter = iterkeys(vrs)
            vardata = hash_to_variable_map[advance_iterator(variable_hash_iter)]
            self._referenced_variable_ids.add(id(vardata))
            var1 = self_variable_symbol_map.getSymbol(vardata)
            if len(vrs)==2:
                vardata = hash_to_variable_map[advance_iterator(variable_hash_iter)]
                self._referenced_variable_ids.add(id(vardata))
                var2 = self_variable_symbol_map.getSymbol(vardata)
            else:
                var2 = var1

            if as_triples is True:
                triples.append((var1, var2, is_obj*coeff))
            else:
                variables1.append(var1)
                variables2.append(var2)
                coefficients.append(coeff)

        if as_triples is True:
            return triples
        else:
            expr = cplex.SparseTriple(ind1=variables1,
                                      ind2=variables2,
                                      val=coefficients)
            return expr

    #
    # method to populate the CPLEX problem instance (interface) from
    # the supplied Pyomo problem instance.
    #
    def _populate_cplex_instance(self, pyomo_instance):
        from pyomo.core.base import Var, Objective, Constraint, SOSConstraint
        from pyomo.repn import canonical_is_constant
        from pyomo.repn import LinearCanonicalRepn, canonical_degree

        self._instance = pyomo_instance

        quadratic_constraints = False
        quadratic_objective = False
        used_sos_constraints = False
        cplex_instance = None

        try:
            cplex_instance = cplex.Cplex()
        except CplexError:
            e = sys.exc_info()[1]
            msg = 'Unable to create Cplex model.  Have you installed the Python'\
            '\n       bindings for Cplex?\n\n\tError message: %s'
            print(sys.exc_info()[1])
            raise Exception(msg % e)

        if self._symbolic_solver_labels:
            labeler = TextLabeler()
        else:
            labeler = NumericLabeler('x')
        symbol_map = SymbolMap()
        pyomo_instance.solutions.add_symbol_map(symbol_map)
        self._smap_id = id(symbol_map)

        # we use this when iterating over the constraints because it
        # will have a much smaller hash table, we also use this for
        # the warm start code after it is cleaned to only contain
        # variables referenced in the constraints
        self_variable_symbol_map = self._variable_symbol_map = SymbolMap()
        var_symbol_pairs = []

        # cplex wants the caller to set the problem type, which is (for current
        # purposes) strictly based on variable type counts.
        num_binary_variables = 0
        num_integer_variables = 0
        num_continuous_variables = 0

        # transfer the variables from pyomo to cplex.
        var_names = []
        var_lbs = []
        var_ubs = []
        var_types = []

        self._referenced_variable_ids.clear()

        for var in pyomo_instance.component_data_objects(Var, active=True):

            if var.fixed and not self._output_fixed_variable_bounds:
                # if a variable is fixed, and we're preprocessing
                # fixed variables (as in not outputting them), there
                # is no need to add them to the compiled model.
                continue

            varname = symbol_map.getSymbol( var, labeler )
            var_names.append(symbol_map.getSymbol( var, labeler ))
            var_symbol_pairs.append((var, varname))

            if (var.lb is None) or (var.lb == -infinity):
                var_lbs.append(-cplex.infinity)
            else:
                var_lbs.append(value(var.lb))
            if (var.ub is None) or (var.ub == infinity):
                var_ubs.append(cplex.infinity)
            else:
                var_ubs.append(value(var.ub))
            if var.is_integer():
                var_types.append(cplex_instance.variables.type.integer)
                num_integer_variables += 1
            elif var.is_binary():
                var_types.append(cplex_instance.variables.type.binary)
                num_binary_variables += 1
            elif var.is_continuous():
                var_types.append(cplex_instance.variables.type.continuous)
                num_continuous_variables += 1
            else:
                raise TypeError("Invalid domain type for variable with name '%s'. "
                                "Variable is not continuous, integer, or binary.")

        self_variable_symbol_map.addSymbols(var_symbol_pairs)
        cplex_instance.variables.add(names=var_names,
                                     lb=var_lbs,
                                     ub=var_ubs,
                                     types=var_types)

        # transfer the constraints.
        expressions = []
        senses = []
        rhss = []
        range_values = []
        names = []

        qexpressions = []
        qlinears = []
        qsenses = []
        qrhss = []
        qnames = []

        # The next loop collects the following component types from the model:
        #  - SOSConstraint
        #  - Objective
        #  - Constraint
        sos1 = self._capabilities.sos1
        sos2 = self._capabilities.sos2
        modelSOS = ModelSOS()
        objective_cntr = 0
        for block in pyomo_instance.block_data_objects(active=True):

            gen_obj_canonical_repn = \
                getattr(block, "_gen_obj_canonical_repn", True)
            gen_con_canonical_repn = \
                getattr(block, "_gen_con_canonical_repn", True)
            # Get/Create the ComponentMap for the repn
            if not hasattr(block,'_canonical_repn'):
                block._canonical_repn = ComponentMap()
            block_canonical_repn = block._canonical_repn

            # SOSConstraints
            for soscondata in block.component_data_objects(SOSConstraint,
                                                           active=True,
                                                           descend_into=False):
                level = soscondata.level
                if (level == 1 and not sos1) or \
                   (level == 2 and not sos2) or \
                   (level > 2):
                    raise RuntimeError(
                        "Solver does not support SOS level %s constraints" % (level))
                modelSOS.count_constraint(symbol_map,
                                          labeler,
                                          self_variable_symbol_map,
                                          soscondata)

            # Objective
            for obj_data in block.component_data_objects(Objective,
                                                         active=True,
                                                         descend_into=False):
                objective_cntr += 1
                if objective_cntr > 1:
                    raise ValueError(
                        "Multiple active objectives found on Pyomo instance '%s'. "
                        "Solver '%s' will only handle a single active objective" \
                        % (pyomo_instance.name, self.type))

                if obj_data.is_minimizing():
                    cplex_instance.objective.\
                        set_sense(cplex_instance.objective.sense.minimize)
                else:
                    cplex_instance.objective.\
                        set_sense(cplex_instance.objective.sense.maximize)

                cplex_instance.objective.set_name(symbol_map.getSymbol(obj_data,
                                                                       labeler))

                if gen_obj_canonical_repn:
                    obj_repn = generate_canonical_repn(obj_data.expr)
                    block_canonical_repn[obj_data] = obj_repn
                else:
                    obj_repn = block_canonical_repn[obj_data]

                if (isinstance(obj_repn, LinearCanonicalRepn) and \
                    (obj_repn.linear == None)) or \
                    canonical_is_constant(obj_repn):
                    print("Warning: Constant objective detected, replacing " + \
                          "with a placeholder to prevent solver failure.")

                    cplex_instance.variables.add(lb=[1],
                                                 ub=[1],
                                                 names=["ONE_VAR_CONSTANT"])
                    objective_expression = [("ONE_VAR_CONSTANT",obj_repn.constant)]
                    cplex_instance.objective.set_linear(objective_expression)

                else:

                    if isinstance(obj_repn, LinearCanonicalRepn):
                        objective_expression, offset = \
                            self._encode_constraint_body_linear_specialized(
                                obj_repn,
                                labeler,
                                as_pairs=True)
                        if offset != 0:
                            cplex_instance.variables.add(lb=[1],
                                                         ub=[1],
                                                         names=["ONE_VAR_CONSTANT"])
                            objective_expression.append(("ONE_VAR_CONSTANT",offset))
                        cplex_instance.objective.set_linear(objective_expression)
                    else:
                        #Linear terms
                        if 1 in obj_repn:
                            objective_expression, offset = \
                                self._encode_constraint_body_linear(obj_repn,
                                                                    labeler,
                                                                    as_pairs=True)
                            if offset != 0:
                                cplex_instance.variables.add(lb=[1],
                                                             ub=[1],
                                                             names=["ONE_VAR_CONSTANT"])
                                objective_expression.append(("ONE_VAR_CONSTANT",offset))
                            cplex_instance.objective.set_linear(objective_expression)

                        #Quadratic terms
                        if 2 in obj_repn:
                            quadratic_objective = True
                            objective_expression = \
                                self._encode_constraint_body_quadratic(
                                    obj_repn,
                                    labeler,
                                    as_triples=True,
                                    is_obj=2.0)
                            cplex_instance.objective.\
                                set_quadratic_coefficients(objective_expression)

                        degree = canonical_degree(obj_repn)
                        if (degree is None) or (degree > 2):
                            raise ValueError(
                                "CPLEXDirect plugin does not support general nonlinear "
                                "objective expressions (only linear or quadratic).\n"
                                "Objective: %s" % (obj_data.name))

            # Constraint
            for con in block.component_data_objects(Constraint,
                                                    active=True,
                                                    descend_into=False):

                if (con.lower is None) and \
                   (con.upper is None):
                    continue  # not binding at all, don't bother

                con_repn = None
                if isinstance(con, LinearCanonicalRepn):
                    con_repn = con
                else:
                    if gen_con_canonical_repn:
                        con_repn = generate_canonical_repn(con.body)
                        block_canonical_repn[con] = con_repn
                    else:
                        con_repn = block_canonical_repn[con]

                # There are conditions, e.g., when fixing variables, under which
                # a constraint block might be empty.  Ignore these, for both
                # practical reasons and the fact that the CPLEX LP format
                # requires a variable in the constraint body.  It is also
                # possible that the body of the constraint consists of only a
                # constant, in which case the "variable" of
                if isinstance(con_repn, LinearCanonicalRepn):
                    if (con_repn.linear is None) and \
                       self._skip_trivial_constraints:
                       continue
                else:
                    # we shouldn't come across a constant canonical repn
                    # that is not LinearCanonicalRepn
                    assert not canonical_is_constant(con_repn)

                name = symbol_map.getSymbol(con, labeler)
                expr=None
                qexpr=None
                quadratic = False
                if isinstance(con_repn, LinearCanonicalRepn):
                    expr, offset = \
                        self._encode_constraint_body_linear_specialized(con_repn,
                                                                        labeler)
                else:
                    degree = canonical_degree(con_repn)
                    if degree == 2:
                        quadratic = True
                    elif (degree != 0) or (degree != 1):
                        raise ValueError(
                            "CPLEXDirect plugin does not support general nonlinear "
                            "constraint expressions (only linear or quadratic).\n"
                            "Constraint: %s" % (con.name))
                    expr, offset = self._encode_constraint_body_linear(con_repn,
                                                                       labeler)

                #Quadratic constraints
                if quadratic:
                    if expr is None:
                        expr = cplex.SparsePair(ind=[0],val=[0.0])
                    quadratic_constraints = True

                    qexpr = self._encode_constraint_body_quadratic(con_repn, labeler)
                    qnames.append(name)

                    if con.equality:
                        # equality constraint.
                        qsenses.append('E')
                        qrhss.append(self._get_bound(con.lower) - offset)

                    elif (con.lower is not None) and (con.upper is not None):
                        raise RuntimeError(
                            "The CPLEXDirect plugin can not translate range "
                            "constraints containing quadratic expressions.")

                    elif con.lower is not None:
                        assert con.upper is None
                        qsenses.append('G')
                        qrhss.append(self._get_bound(con.lower) - offset)

                    else:
                        qsenses.append('L')
                        qrhss.append(self._get_bound(con.upper) - offset)

                    qlinears.append(expr)
                    qexpressions.append(qexpr)

                else:
                    names.append(name)
                    expressions.append(expr)

                    if con.equality:
                        # equality constraint.
                        senses.append('E')
                        rhss.append(self._get_bound(con.lower) - offset)
                        range_values.append(0.0)

                    elif (con.lower is not None) and (con.upper is not None):
                        # ranged constraint.
                        senses.append('R')
                        lower_bound = self._get_bound(con.lower) - offset
                        upper_bound = self._get_bound(con.upper) - offset
                        rhss.append(lower_bound)
                        range_values.append(upper_bound - lower_bound)

                    elif con.lower is not None:
                        senses.append('G')
                        rhss.append(self._get_bound(con.lower) - offset)
                        range_values.append(0.0)

                    else:
                        senses.append('L')
                        rhss.append(self._get_bound(con.upper) - offset)
                        range_values.append(0.0)

        if modelSOS.sosType:
            for key in modelSOS.sosType:
                cplex_instance.SOS.add(type = modelSOS.sosType[key],
                                       name = modelSOS.sosName[key],
                                       SOS = [modelSOS.varnames[key],
                                              modelSOS.weights[key]])
                self._referenced_variable_ids.update(modelSOS.varids[key])
            used_sos_constraints = True

        fixed_upper_bounds = []
        fixed_lower_bounds = []
        for var_id in self._referenced_variable_ids:
            varname = self._variable_symbol_map.byObject[var_id]
            vardata = self._variable_symbol_map.bySymbol[varname]()
            if vardata.fixed:
                if not self._output_fixed_variable_bounds:
                    raise ValueError(
                        "Encountered a fixed variable (%s) inside an active objective"
                        " or constraint expression on model %s, which is usually "
                        "indicative of a preprocessing error. Use the IO-option "
                        "'output_fixed_variable_bounds=True' to suppress this error "
                        "and fix the variable by overwriting its bounds in the Cplex "
                        "instance." % (vardata.name,pyomo_instance.name))

                fixed_lower_bounds.append((varname,vardata.value))
                fixed_upper_bounds.append((varname,vardata.value))

        if len(fixed_upper_bounds):
            cplex_instance.variables.set_upper_bounds(fixed_upper_bounds)
        if len(fixed_lower_bounds):
            cplex_instance.variables.set_upper_bounds(fixed_lower_bounds)

        cplex_instance.linear_constraints.add(lin_expr=expressions,
                                              senses=senses,
                                              rhs=rhss,
                                              range_values=range_values,
                                              names=names)

        for index in xrange(len(qexpressions)):
            cplex_instance.quadratic_constraints.add(lin_expr=qlinears[index],
                                                     quad_expr=qexpressions[index],
                                                     sense=qsenses[index],
                                                     rhs=qrhss[index],
                                                     name=qnames[index])

        # This gets rid of the annoying "Freeing MIP data." message.
        def _filter_freeing_mip_data(val):
            if val.strip() == 'Freeing MIP data.':
                return ""
            return val
        cplex_instance.set_warning_stream(sys.stderr,
                                          fn=_filter_freeing_mip_data)

        # set the problem type based on the variable counts.
        if (quadratic_objective is True) or (quadratic_constraints is True):
            if (num_integer_variables > 0) or \
               (num_binary_variables > 0) or \
               (used_sos_constraints):
                if quadratic_constraints is True:
                    cplex_instance.set_problem_type(
                        cplex_instance.problem_type.MIQCP)
                else:
                    cplex_instance.set_problem_type(
                        cplex_instance.problem_type.MIQP)
            else:
                if quadratic_constraints is True:
                    cplex_instance.set_problem_type(
                        cplex_instance.problem_type.QCP)
                else:
                    cplex_instance.set_problem_type(
                        cplex_instance.problem_type.QP)
        elif (num_integer_variables > 0) or \
             (num_binary_variables > 0) or \
             (used_sos_constraints):
            cplex_instance.set_problem_type(
                cplex_instance.problem_type.MILP)
        else:
            cplex_instance.set_problem_type(
                cplex_instance.problem_type.LP)

        # restore the warning stream without our filter function
        cplex_instance.set_warning_stream(sys.stderr)

        self._active_cplex_instance = cplex_instance

    #
    # cplex has a simple, easy-to-use warm-start capability.
    #
    def warm_start_capable(self):
        return True

    #
    # write a warm-start file in the CPLEX MST format.
    #
    def _warm_start(self, instance):

        # the iteration order is identical to that used in generating
        # the cplex instance, so all should be well.
        variable_names = []
        variable_values = []

        for symbol, vardata in iteritems(self._variable_symbol_map.bySymbol):
            if vardata().value is not None:
                variable_names.append(symbol)
                variable_values.append(vardata().value)

        if len(variable_names):
            self._active_cplex_instance.MIP_starts.add(
                [variable_names, variable_values],
                self._active_cplex_instance.MIP_starts.effort_level.auto)

    def _default_results_format(self, prob_format):
        return None

    # over-ride presolve to extract the warm-start keyword, if specified.
    def _presolve(self, *args, **kwds):

        from pyomo.core.base.PyomoModel import Model

        # create a context in the temporary file manager for
        # this plugin - is "pop"ed in the _postsolve method.
        pyutilib.services.TempfileManager.push()

        self._warm_start_solve = kwds.pop('warmstart', False)
        self._keepfiles = kwds.pop('keepfiles', False)
        # extract io_options here as well, since there is
        # way to tell what kwds were consumed inside
        # OptSolver._presolve. It will be up to that method
        # to decide if remaining kwds are error worthy
        self._symbolic_solver_labels = \
            kwds.pop('symbolic_solver_labels', False)
        self._output_fixed_variable_bounds = \
            kwds.pop('output_fixed_variable_bounds', False)
        # Skip writing constraints whose body section is fixed (i.e., no variables)
        self._skip_trivial_constraints = \
            kwds.pop("skip_trivial_constraints", False)
        # TODO: A bad name for it here, but possibly still
        #       useful (perhaps generalize the name)
        #self._file_determinism = \
        #    kwds.pop('file_determinism', 1)

        # this implies we have a custom solution "parser",
        # preventing the OptSolver _presolve method from
        # creating one
        self._results_format = ResultsFormat.soln
        # use the base class _presolve to consume the
        # important keywords
        OptSolver._presolve(self, *args, **kwds)

        if self._log_file is None:
            self._log_file = pyutilib.services.TempfileManager.\
                             create_tempfile(suffix = '.cplex.log')

        # Possible TODOs
        if self._timelimit is not None:
            logger.warning("The 'timelimit' keyword will be ignored "
                           "for solver="+self.type)
        if self._soln_file is not None:
            logger.warning("The 'soln_file' keyword will be ignored "
                           "for solver="+self.type)

        self.available()

        # Step 1: extract the pyomo instance from the input arguments,
        #         cache it, and create the corresponding (as of now empty)
        #         CPLEX problem instance.
        if len(args) != 1:
            msg = "The CPLEXDirect plugin method '_presolve' must be supplied "\
                  "a single problem instance - %s were supplied"
            raise ValueError(msg % len(args))

        model = args[ 0 ]
        if not isinstance(model, Model):
            msg = "The problem instance supplied to the CPLEXDirect plugin " \
                  "method '_presolve' must be of type 'Model' - "\
                  "interface does not currently support file names"
            raise ValueError(msg)

        # TBD-document.
        self._populate_cplex_instance(model)

        # Clean up the symbol map to only contain variables referenced
        # in the constraints **NOTE**: The warmstart method (if called
        # below), relies on a "clean" symbol map
        vars_to_delete = set(self._variable_symbol_map.byObject.keys()) - \
                         self._referenced_variable_ids
        sm_byObject = model.solutions.symbol_map[self._smap_id].byObject
        sm_bySymbol = model.solutions.symbol_map[self._smap_id].bySymbol
        #sm_bySymbol = self._symbol_map.bySymbol
        assert(len(model.solutions.symbol_map[self._smap_id].aliases) == 0)
        var_sm_byObject = self._variable_symbol_map.byObject
        var_sm_bySymbol = self._variable_symbol_map.bySymbol
        for varid in vars_to_delete:
            symbol = var_sm_byObject[varid]
            del sm_byObject[varid]
            del sm_bySymbol[symbol]
            del var_sm_byObject[varid]
            del var_sm_bySymbol[symbol]

        if 'write' in self.options:
            fname = self.options.write
            self._active_cplex_instance.write(fname)

        # Handle other keywords

        # if the first argument is a string (representing a filename),
        # then we don't have an instance => the solver is being applied
        # to a file.

        # FIXME: This appears to be a bogus test: we raise an exception
        # above if len(args) != 1 or type(args[0]) != Model
        if (len(args) > 0) and not isinstance(model, basestring):

            # write the warm-start file - currently only supports MIPs.
            # we only know how to deal with a single problem instance.
            if self._warm_start_solve:

                if len(args) != 1:
                    msg = "CPLEX _presolve method can only handle a single " \
                          "problem instance - %s were supplied"
                    raise ValueError(msg % len(args))

                cplex_instance = self._active_cplex_instance
                cplex_problem_type = cplex_instance.get_problem_type()
                if (cplex_problem_type == cplex_instance.problem_type.MILP) or \
                   (cplex_problem_type == cplex_instance.problem_type.MIQP) or \
                   (cplex_problem_type == cplex_instance.problem_type.MIQCP):
                    start_time = time.time()
                    self._warm_start(model)
                    end_time = time.time()
                    if self._report_timing is True:
                        print("Warm start write time=%.2f seconds"
                              % (end_time-start_time))

    #
    # TBD
    #
    def _apply_solver(self):

        # set up all user-specified parameters.
        if (self.options.mipgap is not None) and (self.options.mipgap > 0.0):
            self._active_cplex_instance.parameters.mip.\
                tolerances.mipgap.set(self.options.mipgap)

        for key in self.options:
            if key == 'relax_integrality' or key == 'mipgap' or key == 'write':
                continue
            else:
                opt_cmd = self._active_cplex_instance.parameters
                key_pieces = key.split('_')
                for key_piece in key_pieces:
                    opt_cmd = getattr(opt_cmd,key_piece)
                opt_cmd.set(self.options[key])

        if 'relax_integrality' in self.options:
            self._active_cplex_instance.set_problem_type(
                self._active_cplex_instance.problem_type.LP)

        if self._tee:
            def _process_stream(arg):
                sys.stdout.write(arg)
                return arg
            self._active_cplex_instance.set_results_stream(
                self._log_file,
                _process_stream)
        else:
            self._active_cplex_instance.set_results_stream(
                self._log_file)

        if self._keepfiles:
            print("Solver log file: "+self._log_file)

        #
        # Kick off the solve.
        #

        # apparently some versions of the CPLEX Python bindings do not
        # have the get_time - so check before accessing.
        if hasattr(self._active_cplex_instance, "get_time"):
            solve_start_time = self._active_cplex_instance.get_time()
            self._active_cplex_instance.solve()
            self._solve_user_time = \
                self._active_cplex_instance.get_time() - solve_start_time
        else:
            self._active_cplex_instance.solve()
            self._solve_user_time = None

        # FIXME: can we get a return code indicating if CPLEX had a
        # significant failure?
        return Bunch(rc=None, log=None)

    def _postsolve(self):

        # the only suffixes that we extract from CPLEX are
        # constraint duals, constraint slacks, and variable
        # reduced-costs. scan through the solver suffix list
        # and throw an exception if the user has specified
        # any others.
        extract_duals = False
        extract_slacks = False
        extract_reduced_costs = False
        for suffix in self._suffixes:
            flag=False
            if re.match(suffix,"dual"):
                extract_duals = True
                flag=True
            if re.match(suffix,"slack"):
                extract_slacks = True
                flag=True
            if re.match(suffix,"rc"):
                extract_reduced_costs = True
                flag=True
            if not flag:
                raise RuntimeError(
                    "***The CPLEXDirect solver plugin cannot "
                    "extract solution suffix="+suffix)

        cplex_instance = self._active_cplex_instance

        if cplex_instance.get_problem_type() in [cplex_instance.problem_type.MILP,
                                                 cplex_instance.problem_type.MIQP,
                                                 cplex_instance.problem_type.MIQCP]:
            extract_reduced_costs = False
            extract_duals = False

        # Remove variables whose absolute value is smaller than
        # CPLEX's epsilon from the results data
        #cplex_instance.cleanup()

        results = SolverResults()
        results.problem.name = cplex_instance.get_problem_name()
        results.problem.lower_bound = None #cplex_instance.solution.
        results.problem.upper_bound = None
        results.problem.number_of_variables = cplex_instance.variables.get_num()
        results.problem.number_of_constraints = \
            cplex_instance.linear_constraints.get_num() \
            + cplex_instance.quadratic_constraints.get_num() \
            + cplex_instance.indicator_constraints.get_num() \
            + cplex_instance.SOS.get_num()
        results.problem.number_of_nonzeros = None
        results.problem.number_of_binary_variables = \
            cplex_instance.variables.get_num_binary()
        results.problem.number_of_integer_variables = \
            cplex_instance.variables.get_num_integer()
        results.problem.number_of_continuous_variables = \
            cplex_instance.variables.get_num() \
            - cplex_instance.variables.get_num_binary() \
            - cplex_instance.variables.get_num_integer() \
            - cplex_instance.variables.get_num_semiinteger()
        #TODO: Does this double-count semi-integers?
        #Should we also remove semi-continuous?
        results.problem.number_of_objectives = 1

        results.solver.name = "CPLEX "+cplex_instance.get_version()
#        results.solver.status = None
        results.solver.return_code = None
        results.solver.message = None
        results.solver.user_time = self._solve_user_time
        results.solver.system_time = None
        results.solver.wallclock_time = None
        results.solver.termination_message = None

        soln = Solution()
        soln_variable = soln.variable
        soln_constraint = soln.constraint

        soln.gap = None # until proven otherwise

        #Get solution status -- for now, if CPLEX returns anything we
        #don't recognize, mark as an error
        soln_status = cplex_instance.solution.get_status()
        if soln_status in [1, 101, 102]:
            results.solver.termination_condition = TerminationCondition.optimal
            soln.status = SolutionStatus.optimal
        elif soln_status in [2, 4, 118, 119]:
            # Note: soln_status of 4 means infeasible or unbounded
            #       and 119 means MIP infeasible or unbounded
            results.solver.termination_condition = TerminationCondition.unbounded
            soln.status = SolutionStatus.unbounded
        elif soln_status in [3, 103]:
            results.solver.termination_condition = TerminationCondition.infeasible
            soln.status = SolutionStatus.infeasible
        else:
            soln.status = SolutionStatus.error

        if cplex_instance.get_problem_type() in [cplex_instance.problem_type.MILP,
                                                 cplex_instance.problem_type.MIQP,
                                                 cplex_instance.problem_type.MIQCP]:
            try:
                upper_bound = cplex_instance.solution.get_objective_value()
                lower_bound = cplex_instance.solution.MIP.get_best_objective() # improperly named, IM(JPW)HO.
                relative_gap = cplex_instance.solution.MIP.get_mip_relative_gap()
                absolute_gap = upper_bound - lower_bound
                soln.gap = absolute_gap
            except CplexSolverError:
                # something went wrong during the solve and no solution
                # exists
                pass

        #Only try to get objective and variable values if a solution exists
        soln_type = cplex_instance.solution.get_solution_type()
        if soln_type > 0:
            soln.objective[cplex_instance.objective.get_name()] = \
                {'Value': cplex_instance.solution.get_objective_value()}
            num_variables = cplex_instance.variables.get_num()
            variable_names = cplex_instance.variables.get_names()
            variable_values = cplex_instance.solution.get_values()
            for i in xrange(num_variables):
                variable_name = variable_names[i]
                soln_variable[variable_name] = {"Value" : variable_values[i]}

            if extract_reduced_costs:
                # get variable reduced costs
                rc_values = cplex_instance.solution.get_reduced_costs()
                for i in xrange(num_variables):
                    soln_variable[variable_names[i]]["Rc"] = rc_values[i]

            if extract_slacks or extract_duals:

                num_linear_constraints = cplex_instance.linear_constraints.get_num()
                constraint_names = cplex_instance.linear_constraints.get_names()

                num_quadratic_constraints = cplex_instance.quadratic_constraints.get_num()
                q_constraint_names = cplex_instance.quadratic_constraints.get_names()

                for i in xrange(num_linear_constraints):
                    soln_constraint[constraint_names[i]] = {}

            if extract_duals:
                # get duals (linear constraints only)
                dual_values = cplex_instance.solution.get_dual_values()
                for i in xrange(num_linear_constraints):
                    soln_constraint[constraint_names[i]]["Dual"] = dual_values[i]

                # CPLEX PYTHON API DOES NOT SUPPORT QUADRATIC DUAL COLLECTION

            if extract_slacks:
                # get linear slacks
                slack_values = cplex_instance.solution.get_linear_slacks()
                for i in xrange(num_linear_constraints):
                    # if both U and L exist (i.e., a range constraint) then
                    # R_ = U-L
                    R_ = cplex_instance.linear_constraints.get_range_values(i)
                    if R_ == 0.0:
                        soln_constraint[constraint_names[i]]["Slack"] = slack_values[i]
                    else:
                        # This is a range constraint for which cplex
                        # always returns the value of f(x)-L. In the
                        # spirit of conforming with the other writer,
                        # I will return the max (in absolute value) of
                        # L-f(x) and U-f(x)
                        Ls_ = slack_values[i]
                        Us_ = R_ - slack_values[i]
                        if Us_ > Ls_:
                            soln_constraint[constraint_names[i]]["Slack"] = Us_
                        else:
                            soln_constraint[constraint_names[i]]["Slack"] = -Ls_

                # get quadratic slacks
                slack_values = cplex_instance.solution.get_quadratic_slacks()
                for i in xrange(num_quadratic_constraints):
                    # if both U and L exist (i.e., a range constraint) then
                    # R_ = U-L
                    soln_constraint[q_constraint_names[i]] = \
                        {"Slack" : slack_values[i]}

            byObject = self._instance.solutions.symbol_map[self._smap_id].byObject
            referenced_varnames = set(byObject[varid]
                                      for varid in self._referenced_variable_ids)
            names_to_delete = set(soln_variable.keys())-referenced_varnames
            for varname in names_to_delete:
                del soln_variable[varname]

            results.solution.insert(soln)

        self.results = results

        # don't know if any of this is necessary!

        # take care of the annoying (and empty) CPLEX temporary files in
        # the current directory.  this approach doesn't seem overly
        # efficient, but python os module functions don't accept regular
        # expression directly.
        try:
            filename_list = glob.glob("cplex.log") + \
                            glob.glob("clone*.log")
            clone_re = re.compile('clone\d+\.log')
            for filename in filename_list:
                # CPLEX temporary files come in two flavors - cplex.log and
                # clone*.log.  the latter is the case for multi-processor
                # environments.
                #
                # IMPT: trap the possible exception raised by the file not existing.
                #       this can occur in pyro environments where > 1 workers are
                #       running CPLEX, and were started from the same directory.
                #       these logs don't matter anyway (we redirect everything),
                #       and are largely an annoyance.
                try:
                    if filename == 'cplex.log':
                        os.remove(filename)
                    elif clone_re.match(filename):
                        os.remove(filename)
                except OSError:
                    pass
        except OSError:
            pass

        self._active_cplex_instance = None
        self._variable_symbol_map = None
        self._instance = None

        # finally, clean any temporary files registered with the temp file
        # manager, created populated *directly* by this plugin.
        pyutilib.services.TempfileManager.pop(remove=not self._keepfiles)

        # let the base class deal with returning results.
        return OptSolver._postsolve(self)

    def _initialize_callbacks(self, model):
        #
        # Called from OptSolver
        #
        cplex_callback = {
            "node-callback":        cplex.callbacks.NodeCallback,
            "solve-callback":       cplex.callbacks.SolveCallback,
            "branch-callback":      cplex.callbacks.BranchCallback,
            "heuristic-callback":   cplex.callbacks.HeuristicCallback,
            "incumbent-callback":   cplex.callbacks.IncumbentCallback,
            "cut-callback":         cplex.callbacks.UserCutCallback,
            "lazycut-callback":     cplex.callbacks.LazyConstraintCallback,
            "crossover-callback":   cplex.callbacks.CrossoverCallback,
            "barrier-callback":     cplex.callbacks.BarrierCallback,
            "simplex-callback":     cplex.callbacks.SimplexCallback,
            "presolve-callback":    cplex.callbacks.PresolveCallback,
            "tuning-callback":      cplex.callbacks.TuningCallback
            }
        #
        for name in self._callback:
            try:
                cb_class = cplex_callback[name]
            except KeyError:
                raise ValueError("Unknown callback name: %s" % name)
            #
            def call_fn(self, *args, **kwds):
                try:
                    self.solver = CplexSolverWrapper(self)
                    self._callback[self.name](self.solver, model)
                except Exception(e):
                    # Should we raise this exception?
                    print("ERROR: "+str(e))
            CallbackClass = type('CallbackClass_'+name.replace('-','_'),
                                 (cb_class,object),
                                 {"_callback":self._callback,
                                  "name":name,
                                  "__call__":call_fn})
            self._active_cplex_instance.register_callback(CallbackClass)

if cplex_import_available is False:
    SolverFactory().deactivate('_cplex_direct')
    SolverFactory().deactivate('_mock_cplexdirect')
