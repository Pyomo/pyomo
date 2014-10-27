#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2010 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________


import logging
import os
import sys
import re
import string
import xml.dom.minidom
import time
import math

from six import itervalues, iterkeys, iteritems, advance_iterator
from six.moves import xrange

logger = logging.getLogger('pyomo.solvers')

_cplex_version = None
try:
    import cplex
    from cplex.exceptions import CplexError
    # create a version tuple of length 4
    _cplex_version = tuple(int(i) for i in cplex.Cplex().get_version().split('.'))
    while(len(_cplex_version) < 4):
        _cplex_version += (0,)
    _cplex_version = _cplex_version[:4]
    cplex_import_available=True
except ImportError:
    cplex_import_available=False

import pyutilib.services
import pyutilib.common
from pyutilib.misc import Bunch, Options

import pyomo.util.plugin
from pyomo.opt.base import *
from pyomo.opt.base.solvers import _extract_version
from pyomo.opt.results import *
from pyomo.opt.solver import *
from pyomo.core.base import SymbolMap, BasicSymbolMap, NumericLabeler, ComponentMap, TextLabeler
from pyomo.core.base.numvalue import value
from pyomo.core.base.block import active_components, active_components_data
from pyomo.solvers import wrappers

try:
    unicode
except:
    basestring = str


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

        sos_items = soscondata.get_items()
        level = soscondata.get_level()

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
                                   "in order to proceed" % (soscondata.cname(True), vardata.cname(True)))
            varids.append(id(vardata))
            varnames.append(variable_symbol_map.getSymbol(vardata))
            weights.append(weight)


class CPLEXDirect(OptSolver):
    """The CPLEX LP/MIP solver
    """

    pyomo.util.plugin.alias('_cplex_direct',  doc='Direct Python interface to the CPLEX LP/MIP solver')

    def __init__(self, **kwds):
        #
        # Call base class constructor
        #
        kwds['type'] = 'cplexdirect'
        OptSolver.__init__(self, **kwds)

        # NOTE: eventually both of the following attributes should be migrated to a common base class.
        # is the current solve warm-started? a transient data member to communicate state information
        # across the _presolve, _apply_solver, and _postsolve methods.
        self.warm_start_solve = False

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

        #
        self.allow_callbacks = True

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
                raise ApplicationError("No CPLEX <-> Python bindings available - CPLEX direct solver functionality is not available")
            else:
                return True

    def version(self):
        if _cplex_version is None:
            return _extract_version('')
        return _cplex_version

    #
    # TBD
    #
    def _evaluate_bound(self, exp):

        from pyomo.core.base import expr

        if exp.is_fixed():
            return exp()
        else:
            raise ValueError("ERROR: non-fixed bound: " + str(exp))

    #
    # CPLEX requires objective expressions to be specified via something other than a sparse pair!
    # NOTE: The returned offset is guaranteed to be a float.
    #
    def _encode_constraint_body_linear(self, expression, labeler, as_pairs=False):

        variables = [] # string names of variables
        coefficients = [] # variable coefficients

        pairs = []

        hash_to_variable_map = expression[-1]
        self_variable_symbol_map = self._variable_symbol_map

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
    # CPLEX requires objective expressions to be specified via something other than a sparse pair!
    # NOTE: The returned offset is guaranteed to be a float.
    # NOTE: This function is a variant of the above, specialized for LinearCanonicalRepn objects.
    #
    def _encode_constraint_body_linear_specialized(self, linear_repn, labeler, as_pairs=False):

        variables = [] # string names of variables
        coefficients = [] # variable coefficients

        pairs = []

        self_variable_symbol_map = self._variable_symbol_map
        for i in xrange(0,len(linear_repn.linear)):

            var_coefficient = linear_repn.linear[i]
            var_value = linear_repn.variables[i]
            self._referenced_variable_ids.add(id(var_value))
            variable_name = self_variable_symbol_map.getSymbol(var_value)
            
            if as_pairs is True:
                pairs.append((variable_name, var_coefficient))
            else:
                variables.append(variable_name)
                coefficients.append(var_coefficient)

        offset=0.0
        if linear_repn.constant != None:
            offset = linear_repn.constant

        if as_pairs is True:
            return pairs, offset
        else:
            expr = cplex.SparsePair(ind=variables, val=coefficients)
            return expr, offset

    #
    #Handle quadratic constraints and objectives
    #
    def _encode_constraint_body_quadratic(self, expression, labeler, as_triples=False, is_obj=1.0):

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
            expr = cplex.SparseTriple(ind1=variables1,ind2=variables2,val=coefficients)
            return expr


    #
    # method to populate the CPLEX problem instance (interface) from the supplied Pyomo problem instance.
    #
    def _populate_cplex_instance(self, pyomo_instance):

        from pyomo.core.base import Var, Objective, Constraint, IntegerSet, BooleanSet, SOSConstraint
        from pyomo.core.base.objective import minimize, maximize
        from pyomo.repn import canonical_is_constant
        from pyomo.repn import LinearCanonicalRepn
    
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


        if self.symbolic_solver_labels is True:
            labeler = TextLabeler()
        else:
            labeler = NumericLabeler('x')
        self_symbol_map = self._symbol_map = SymbolMap(pyomo_instance)
        # we use this when iterating over the constraints because it will have a much smaller hash
        # table, we also use this for the warm start code after it is cleaned to only contain
        # variables referenced in the constraints
        self_variable_symbol_map = self._variable_symbol_map = BasicSymbolMap()
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

        for block in pyomo_instance.all_blocks():
            for var in active_components_data(block,Var):

                varname = self_symbol_map.getSymbol( var, labeler )
                var_names.append(self_symbol_map.getSymbol( var, labeler ))
                var_symbol_pairs.append((var, varname))

                if var.lb is None:
                    var_lbs.append(-cplex.infinity)
                else:
                    var_lbs.append(value(var.lb))
                if var.ub is None:
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

        self_variable_symbol_map.updateSymbols(var_symbol_pairs)
        cplex_instance.variables.add(names=var_names, lb=var_lbs, ub=var_ubs, types=var_types)
        
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
        for block in pyomo_instance.all_blocks():

            block_canonical_repn = getattr(block,"canonical_repn",None)
            if block_canonical_repn is None:
                raise ValueError("No canonical_repn ComponentMap was found on "
                                 "block with name %s. Did you forget to preprocess?"
                                 % (block.cname(True)))

            # SOSConstraints
            for soscondata in active_components_data(block,SOSConstraint):
                level = soscondata.get_level()
                if (level == 1 and not sos1) or (level == 2 and not sos2) or (level > 2):
                    raise Exception("Solver does not support SOS level %s constraints" % (level,))
                modelSOS.count_constraint(self_symbol_map,
                                          labeler,
                                          self_variable_symbol_map,
                                          soscondata)

            # Objective
            for obj_data in active_components_data(block,Objective):

                objective_cntr += 1
                if objective_cntr > 1:
                    raise ValueError("Multiple active objectives found on Pyomo instance '%s'. "
                                     "Solver '%s' will only handle a single active objective" \
                                     % (pyomo_instance.cname(True), self.type))

                if obj_data.is_minimizing():
                    cplex_instance.objective.set_sense(cplex_instance.objective.sense.minimize)
                else:
                    cplex_instance.objective.set_sense(cplex_instance.objective.sense.maximize)

                cplex_instance.objective.set_name(self_symbol_map.getSymbol(obj_data, labeler))
            
                obj_repn = block_canonical_repn.get(obj_data)
                if obj_repn is None:
                    raise ValueError("No entry found in canonical_repn ComponentMap on "
                                     "block %s for active objective with name %s. "
                                     "Did you forget to preprocess?"
                                     % (block.cname(True), obj_data.cname(True)))

                if (isinstance(obj_repn, LinearCanonicalRepn) and (obj_repn.linear == None)) or canonical_is_constant(obj_repn):
                    print("Warning: Constant objective detected, replacing " + \
                          "with a placeholder to prevent solver failure.")

                    cplex_instance.variables.add(lb=[1],ub=[1],names=["ONE_VAR_CONSTANT"])
                    objective_expression = [("ONE_VAR_CONSTANT",obj_repn.constant)]
                    cplex_instance.objective.set_linear(objective_expression)
                
                else:

                    if isinstance(obj_repn, LinearCanonicalRepn):
                        objective_expression, offset = self._encode_constraint_body_linear_specialized(obj_repn,
                                                                                                       labeler,
                                                                                                       as_pairs=True)
                        if offset != 0:
                            cplex_instance.variables.add(lb=[1],ub=[1],names=["ONE_VAR_CONSTANT"])
                            objective_expression.append(("ONE_VAR_CONSTANT",offset))
                        cplex_instance.objective.set_linear(objective_expression)
                    else:
                        #Linear terms
                        if 1 in obj_repn:
                            objective_expression, offset = self._encode_constraint_body_linear(obj_repn,
                                                                                               labeler,
                                                                                               as_pairs=True)
                            if offset != 0:
                                cplex_instance.variables.add(lb=[1],ub=[1],names=["ONE_VAR_CONSTANT"])
                                objective_expression.append(("ONE_VAR_CONSTANT",offset))
                            cplex_instance.objective.set_linear(objective_expression)

                        #Quadratic terms 
                        if 2 in obj_repn:
                            quadratic_objective = True
                            objective_expression = self._encode_constraint_body_quadratic(obj_repn, 
                                                                                          labeler, 
                                                                                          as_triples=True, 
                                                                                          is_obj=2.0)
                            cplex_instance.objective.set_quadratic_coefficients(objective_expression)

            # Constraint
            for constraint in active_components(block, Constraint):
                if constraint.trivial:
                    continue

                for con in itervalues(constraint): # TBD: more efficient looping here.
                    if not con.active:
                        continue

                    con_repn = block_canonical_repn.get(con)
                    if con_repn is None:
                        raise ValueError("No entry found in canonical_repn ComponentMap on "
                                         "block %s for active constraint with name %s. "
                                         "Did you forget to preprocess?"
                                         % (block.cname(True), con.cname(True)))

                    # There are conditions, e.g., when fixing variables, under which
                    # a constraint block might be empty.  Ignore these, for both
                    # practical reasons and the fact that the CPLEX LP format
                    # requires a variable in the constraint body.  It is also
                    # possible that the body of the constraint consists of only a
                    # constant, in which case the "variable" of
                    if isinstance(con_repn, LinearCanonicalRepn):
                        if con_repn.linear == None:
                           continue
                    else:
                       if canonical_is_constant(con_repn):
                           continue

                    name=self_symbol_map.getSymbol(con,labeler)
                    expr=None
                    qexpr=None

                    #Linear constraints
                    quadratic = False
                    if isinstance(con_repn, LinearCanonicalRepn):
                        expr, offset = self._encode_constraint_body_linear_specialized(con_repn, labeler)
                    elif 2 in con_repn:
                        quadratic=True
                    elif 1 in con_repn:
                        expr, offset = self._encode_constraint_body_linear(con_repn, labeler)

                    #Quadratic constraints
                    if quadratic is True:
                        if expr is None:
                            expr = cplex.SparsePair(ind=[0],val=[0.0])
                        quadratic_constraints = True

                        qexpr = self._encode_constraint_body_quadratic(con_repn,labeler)
                        qnames.append(name)

                        if con._equality:
                            # equality constraint.
                            qsenses.append('E')
                            bound_expr = con.lower
                            bound = self._evaluate_bound(bound_expr)
                            qrhss.append(bound)

                        elif con.lower is not None:
                            assert con.upper is not None
                            qsenses.append('G')
                            bound_expr = con.lower
                            bound = self._evaluate_bound(bound_expr)
                            qrhss.append(bound)

                        else:
                            qsenses.append('L')
                            bound_expr = con.upper
                            bound = self._evaluate_bound(bound_expr)
                            qrhss.append(bound)

                        qlinears.append(expr)
                        qexpressions.append(qexpr)

                    else:
                        names.append(name)
                        expressions.append(expr)

                        if con._equality:
                            # equality constraint.
                            senses.append('E')
                            bound_expr = con.lower
                            bound = self._evaluate_bound(bound_expr) - offset
                            rhss.append(bound)
                            range_values.append(0.0)

                        elif (con.lower is not None) and (con.upper is not None):
                            # ranged constraint.
                            senses.append('R')
                            lower_bound_expr = con.lower # TBD - watch the offset - why not subtract?
                            lower_bound = self._evaluate_bound(lower_bound_expr)
                            upper_bound_expr = con.upper # TBD - watch the offset - why not subtract?
                            upper_bound = self._evaluate_bound(upper_bound_expr)
                            rhss.append(lower_bound)
                            range_values.append(upper_bound-lower_bound)

                        elif con.lower is not None:
                            senses.append('G')
                            bound_expr = con.lower
                            bound = self._evaluate_bound(bound_expr) - offset
                            rhss.append(bound)
                            range_values.append(0.0)

                        else:
                            senses.append('L')
                            bound_expr = con.upper
                            bound = self._evaluate_bound(bound_expr) - offset
                            rhss.append(bound)
                            range_values.append(0.0)

        if modelSOS.sosType:
            for key in modelSOS.sosType:
                cplex_instance.SOS.add(type = modelSOS.sosType[key], \
                                       name = modelSOS.sosName[key], \
                                       SOS = [modelSOS.varnames[key], modelSOS.weights[key]] )
                self._referenced_variable_ids.update(modelSOS.varids[key])
            used_sos_constraints = True

        fixed_upper_bounds = []
        fixed_lower_bounds = []
        for var_id in self._referenced_variable_ids:
            varname = self._variable_symbol_map.byObject[var_id]
            vardata = self._variable_symbol_map.bySymbol[varname]
            if vardata.fixed:
                if not self.output_fixed_variable_bounds:
                    raise ValueError("Encountered a fixed variable (%s) inside an active objective "
                                     "or constraint expression on model %s, which is usually indicative of "
                                     "a preprocessing error. Use the IO-option 'output_fixed_variable_bounds=True' "
                                     "to suppress this error and fix the variable by overwriting its bounds in "
                                     "the Cplex instance."
                                     % (vardata.cname(True),pyomo_instance.cname(True),))

                fixed_lower_bounds.append((varname,vardata.value))
                fixed_upper_bounds.append((varname,vardata.value))

        if len(fixed_upper_bounds):
            cplex_instance.variables.set_upper_bounds(fixed_upper_bounds)
        if len(fixed_lower_bounds):
            cplex_instance.variables.set_upper_bounds(fixed_lower_bounds)

        cplex_instance.linear_constraints.add(lin_expr=expressions, senses=senses, rhs=rhss, range_values=range_values, names=names)

        for index in xrange(len(qexpressions)):
            cplex_instance.quadratic_constraints.add(lin_expr=qlinears[index], quad_expr=qexpressions[index], sense=qsenses[index], rhs=qrhss[index], name=qnames[index])
        
        # set the problem type based on the variable counts.
        if (quadratic_objective is True) or (quadratic_constraints is True):
            if (num_integer_variables > 0) or (num_binary_variables > 0) or (used_sos_constraints):
                if quadratic_constraints is True:
                    cplex_instance.set_problem_type(cplex_instance.problem_type.MIQCP)
                else:
                    cplex_instance.set_problem_type(cplex_instance.problem_type.MIQP)
            else:
                if quadratic_constraints is True:
                    cplex_instance.set_problem_type(cplex_instance.problem_type.QCP)
                else:
                    cplex_instance.set_problem_type(cplex_instance.problem_type.QP)
        elif (num_integer_variables > 0) or (num_binary_variables > 0) or (used_sos_constraints):
            cplex_instance.set_problem_type(cplex_instance.problem_type.MILP)
        else:
            cplex_instance.set_problem_type(cplex_instance.problem_type.LP)

        self._active_cplex_instance = cplex_instance

    #
    # cplex has a simple, easy-to-use warm-start capability.
    #
    def warm_start_capable(self):

        return True

    #
    # write a warm-start file in the CPLEX MST format.
    #
    def warm_start(self, instance):

        # the iteration order is identical to that used in generating
        # the cplex instance, so all should be well.
        variable_names = []
        variable_values = []

        for symbol, vardata in iteritems(self._variable_symbol_map.bySymbol):
            if vardata.value is not None:
                variable_names.append(symbol)
                variable_values.append(vardata.value)

        if len(variable_names):
            self._active_cplex_instance.MIP_starts.add([variable_names, variable_values],
                                                       self._active_cplex_instance.MIP_starts.effort_level.auto)

    # over-ride presolve to extract the warm-start keyword, if specified.
    def _presolve(self, *args, **kwds):

        from pyomo.core.base.var import Var
        from pyomo.core.base.PyomoModel import Model

        self.warm_start_solve = False
        self.keepfiles = False
        self.tee = False
        self.symbolic_solver_labels = False
        self.output_fixed_variable_bounds = False
        for key in kwds:
            ### copied from base class _presolve
            warn = False
            if key == "logfile":
                if kwds[key] is not None:
                    warn = True
            elif key == "solnfile":
                if kwds[key] is not None:
                    warn = True
            elif key == "timelimit":
                if kwds[key] is not None:
                    warn = True
            elif key == "tee":
                self.tee=bool(kwds[key])
            elif key == "options":
                self.set_options(kwds[key])
            elif key == "available":
                self._assert_available=True
            elif key == "symbolic_solver_labels":
                self.symbolic_solver_labels = bool(kwds[key])
            elif key == "output_fixed_variable_bounds":
                self.output_fixed_variable_bounds = bool(kwds[key])
            elif key == "suffixes":
                self.suffixes=kwds[key]
            ###
            elif key == 'keepfiles':
                self.keepfiles = bool(kwds[key])
            elif key == 'warmstart':
                self.warm_start_solve = bool(kwds[key])
            else:
                raise ValueError("Unknown option="+key+" for solver="+self.type)

            if warn is True:
                logger.warn('"'+key+'" keyword ignored by solver='+self.type)

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
        vars_to_delete = set(self._variable_symbol_map.byObject.keys())-self._referenced_variable_ids
        sm_byObject = self._symbol_map.byObject
        sm_bySymbol = self._symbol_map.bySymbol
        assert(len(self._symbol_map.aliases) == 0)
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
        if (len(args) > 0) and not isinstance(model,basestring):

            # write the warm-start file - currently only supports MIPs.
            # we only know how to deal with a single problem instance.
            if self.warm_start_solve is True:

                if len(args) != 1:
                    msg = "CPLEX _presolve method can only handle a single " \
                          "problem instance - %s were supplied"
                    raise ValueError(msg % len(args))
                
                cplex_instance = self._active_cplex_instance
                if cplex_instance.get_problem_type() in (cplex_instance.problem_type.MILP,
                                                         cplex_instance.problem_type.MIQP,
                                                         cplex_instance.problem_type.MIQCP):
                    start_time = time.time()
                    self.warm_start(model)
                    end_time = time.time()
                    if self._report_timing is True:
                        print("Warm start write time=%.2f seconds" % (end_time-start_time))

        # This does not use weak references so we need to release references to external model variables
        del self._variable_symbol_map

    #
    # TBD
    #
    def _apply_solver(self):

        # set up all user-specified parameters.
        if (self.options.mipgap is not None) and (self.options.mipgap > 0.0):
            self._active_cplex_instance.parameters.mip.tolerances.mipgap.set(self.options.mipgap)

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
            self._active_cplex_instance.set_problem_type(self._active_cplex_instance.problem_type.LP)
        
        # and kick off the solve.
        if self.tee == True:
            #Should this use pyutilib's tee_io? I couldn't find where
            #other solvers set output using tee=True/False
            from sys import stdout
            self._active_cplex_instance.set_results_stream(stdout)
        elif self.keepfiles == True:
            log_file = pyutilib.services.TempfileManager.create_tempfile(suffix = '.cplex.log')
            print("Solver log file: " + log_file)
            self._active_cplex_instance.set_results_stream(log_file)
            #Not sure why the following doesn't work. As a result, it's either stream output
            #or write a logfile, but not both.
            #self._active_cplex_instance.set_log_stream(log_file)
        else:
            self._active_cplex_instance.set_results_stream(None)

        # apparently some versions of the CPLEX Python bindings do not
        # have the get_time - so check before accessing.
        if hasattr(self._active_cplex_instance, "get_time"):
            solve_start_time = self._active_cplex_instance.get_time()
            self._active_cplex_instance.solve()
            self._solve_user_time = self._active_cplex_instance.get_time() - solve_start_time
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
        for suffix in self.suffixes:
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
                raise RuntimeError("***The CPLEXDirect solver plugin cannot extract solution suffix="+suffix)

        instance = self._active_cplex_instance

        if instance.get_problem_type() in [instance.problem_type.MILP,
                                           instance.problem_type.MIQP,
                                           instance.problem_type.MIQCP]:
            extract_reduced_costs = False
            extract_duals = False

        # Remove variables whose absolute value is smaller than CPLEX's epsilon from the results data
        #instance.cleanup()

        results = SolverResults()
        results.problem.name = instance.get_problem_name()
        results.problem.lower_bound = None #instance.solution.
        results.problem.upper_bound = None
        results.problem.number_of_variables = instance.variables.get_num()
        results.problem.number_of_constraints = instance.linear_constraints.get_num() \
                                                + instance.quadratic_constraints.get_num() \
                                                + instance.indicator_constraints.get_num() \
                                                + instance.SOS.get_num()
        results.problem.number_of_nonzeros = None
        results.problem.number_of_binary_variables = instance.variables.get_num_binary()
        results.problem.number_of_integer_variables = instance.variables.get_num_integer()
        results.problem.number_of_continuous_variables = instance.variables.get_num() \
                                                            - instance.variables.get_num_binary() \
                                                            - instance.variables.get_num_integer() \
                                                            - instance.variables.get_num_semiinteger()
                                                            #TODO: Does this double-count semi-integers?
                                                            #Should we also remove semi-continuous?
        results.problem.number_of_objectives = 1

        results.solver.name = "CPLEX "+instance.get_version()
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

        #Get solution status -- for now, if CPLEX returns anything we don't recognize, mark as an error
        soln_status = instance.solution.get_status()
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

        # the definition of relative gap in the case of CPLEX MIP is
        # |best - bestinteger| / ((1e-10)+|bestinteger|).  for some
        # reason, the CPLEX Python interface doesn't appear to support
        # extraction of the absolute gap, so we have to compute it.
        m = instance.solution.quality_metric
        try:
            relative_gap = instance.solution.MIP.get_mip_relative_gap()
            best_integer = instance.solution.MIP.get_best_objective()
            diff = relative_gap * (1.0e-10 + math.fabs(best_integer))
            soln.gap = diff 
        except CplexError:
            #
            # If an error occurs, then the model is not a MIP
            #
            pass

        #Only try to get objective and variable values if a solution exists
        soln_type = instance.solution.get_solution_type()
        if soln_type > 0:
            soln.objective[instance.objective.get_name()].value = instance.solution.get_objective_value()
            num_variables = instance.variables.get_num()
            variable_names = instance.variables.get_names()
            variable_values = instance.solution.get_values()
            for i in xrange(num_variables):
                variable_name = variable_names[i]
                soln_variable[variable_name] = {"Value" : variable_values[i]}
            
            if extract_reduced_costs:
                # get variable reduced costs
                rc_values = instance.solution.get_reduced_costs()
                for i in xrange(num_variables):
                    soln_variable[variable_names[i]]["Rc"] = rc_values[i]

            num_linear_constraints = instance.linear_constraints.get_num()
            constraint_names = instance.linear_constraints.get_names()
            
            num_quadratic_constraints = instance.quadratic_constraints.get_num()
            q_constraint_names = instance.quadratic_constraints.get_names()

            if extract_slacks or extract_duals:
                for i in xrange(num_linear_constraints):
                    soln_constraint[constraint_names[i]] = {}
                    
            if extract_duals:
                # get duals (linear constraints only)
                dual_values = instance.solution.get_dual_values()
                for i in xrange(num_linear_constraints):
                    soln_constraint[constraint_names[i]]["Dual"] = dual_values[i]

                # CPLEX PYTHON API DOES NOT SUPPORT QUADRATIC DUAL COLLECTION

            if extract_slacks:
                # get linear slacks
                slack_values = instance.solution.get_linear_slacks() 
                for i in xrange(num_linear_constraints):
                    # if both U and L exist (i.e., a range constraint) then 
                    # R_ = U-L
                    R_ = instance.linear_constraints.get_range_values(i)
                    if R_ == 0.0:
                        soln_constraint[constraint_names[i]]["Slack"] = slack_values[i]
                    else:
                        # This is a range constraint for which cplex always returns the
                        # value of f(x)-L. In the spirit of conforming with the other writer,
                        # I will return the max (in absolute value) of L-f(x) and U-f(x)
                        Ls_ = slack_values[i]
                        Us_ = R_ - slack_values[i]
                        if Us_ > Ls_:
                            soln_constraint[constraint_names[i]]["Slack"] = Us_
                        else:
                            soln_constraint[constraint_names[i]]["Slack"] = -Ls_
                
                # get quadratic slacks
                slack_values = instance.solution.get_quadratic_slacks() 
                for i in xrange(num_quadratic_constraints):
                    # if both U and L exist (i.e., a range constraint) then 
                    # R_ = U-L
                    soln_constraint[q_constraint_names[i]] = {"Slack" : slack_values[i]}
            
            byObject = self._symbol_map.byObject
            referenced_varnames = set(byObject[varid] for varid in self._referenced_variable_ids)
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
        filename_list = os.listdir(".")
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
            CallbackClass = type('CallbackClass_'+name.replace('-','_'), (cb_class,object), {"_callback":self._callback, "name":name, "__call__":call_fn})
            self._active_cplex_instance.register_callback(CallbackClass)


if cplex_import_available is False:
    SolverFactory().deactivate('_cplex_direct')
    SolverFactory().deactivate('_mock_cplexdirect')
