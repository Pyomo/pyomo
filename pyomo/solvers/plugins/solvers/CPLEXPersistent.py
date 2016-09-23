#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import sys
import logging
import os
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
from pyomo.solvers.plugins.solvers.CPLEXDirect import (CPLEXDirect,
                                                       ModelSOS)

logger = logging.getLogger('pyomo.solvers')

try:
    unicode
except:
    basestring = str

try:
    import cplex
    from cplex.exceptions import CplexError, CplexSolverError
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

class CPLEXPersistent(CPLEXDirect, PersistentSolver):
    """The CPLEX LP/MIP solver
    """

    pyomo.util.plugin.alias('_cplex_persistent',
                            doc='Persistent Python interface to the CPLEX LP/MIP solver')

    def __init__(self, **kwds):
        #
        # Call base class constructor
        #
        kwds['type'] = 'cplexpersistent'
        CPLEXDirect.__init__(self, **kwds)

        # maps pyomo var data labels to the corresponding CPLEX variable id.
        self._cplex_variable_ids = {}
        self._cplex_variable_names = None

    #
    # updates all variable bounds in the compiled model - handles
    # fixed variables and related issues.  re-does everything from
    # scratch by default, ignoring whatever was specified
    # previously. if the value associated with the keyword
    # vars_to_update is a non-empty list (assumed to be variable name
    # / index pairs), then only the bounds for those variables are
    # updated.  this function assumes that the variables themselves
    # already exist in the compiled model.
    #
    def compile_variable_bounds(self, pyomo_instance, vars_to_update):

        from pyomo.core.base import Var

        if self._active_cplex_instance is None:
            raise RuntimeError("***The CPLEXPersistent solver plugin "
                               "cannot compile variable bounds - no "
                               "instance is presently compiled")

        # the bound update entries should be name-value pairs
        new_lower_bounds = []
        new_upper_bounds = []

        # operates through side effects on the above lists!
        def update_bounds_lists(var_name):

            var_lb = None
            var_ub = None

            if var_data.fixed and self._output_fixed_variable_bounds:
                var_lb = var_ub = var_data.value
            elif var_data.fixed:
                # if we've been directed to not deal with fixed
                # variables, then skip - they should have been
                # compiled out of any description of the constraints
                return
            else:
                if var_data.lb is None:
                    var_lb = -cplex.infinity
                else:
                    var_lb = value(var_data.lb)

                if var_data.ub is None:
                    var_ub = cplex.infinity
                else:
                    var_ub= value(var_data.ub)

            var_cplex_id = self._cplex_variable_ids[var_name]

            new_lower_bounds.append((var_cplex_id, var_lb))
            new_upper_bounds.append((var_cplex_id, var_ub))

        if len(vars_to_update) == 0:
            for var_data in pyomo_instance.component_data_objects(Var, active=True):
                var_name = self._symbol_map.getSymbol(var_data, self._labeler)
                update_bounds_lists(var_name)
        else:
            for var_name, var_index in vars_to_update:
                var = pyomo_instance.find_component(var_name)
                # TBD - do some error checking!
                var_data = var[var_index]
                var_name = self._symbol_map.getSymbol(var_data, self._labeler)
                update_bounds_lists(var_name)

        self._active_cplex_instance.variables.set_lower_bounds(new_lower_bounds)
        self._active_cplex_instance.variables.set_upper_bounds(new_upper_bounds)

    #
    # method to compile objective of the input pyomo instance.
    # TBD:
    #   it may be smarter just to track the associated pyomo instance,
    #   and re-compile it automatically from a cached local attribute.
    #   this would ensure consistency, among other things!
    #
    def compile_objective(self, pyomo_instance):

        from pyomo.core.base import Objective
        from pyomo.repn import canonical_is_constant, LinearCanonicalRepn, canonical_degree

        if self._active_cplex_instance is None:
            raise RuntimeError("***The CPLEXPersistent solver plugin "
                               "cannot compile objective - no "
                               "instance is presently compiled")

        cplex_instance = self._active_cplex_instance

        cntr = 0
        for block in pyomo_instance.block_data_objects(active=True):
            gen_obj_canonical_repn = \
                getattr(block, "_gen_obj_canonical_repn", True)
            # Get/Create the ComponentMap for the repn
            if not hasattr(block,'_canonical_repn'):
                block._canonical_repn = ComponentMap()
            block_canonical_repn = block._canonical_repn

            for obj_data in block.component_data_objects(Objective,
                                                         active=True,
                                                         descend_into=False):

                cntr += 1
                if cntr > 1:
                    raise ValueError(
                        "Multiple active objectives found on Pyomo instance '%s'. "
                        "Solver '%s' will only handle a single active objective" \
                        % (pyomo_instance.name, self.type))

                if obj_data.is_minimizing():
                    cplex_instance.objective.set_sense(
                        cplex_instance.objective.sense.minimize)
                else:
                    cplex_instance.objective.set_sense(
                        cplex_instance.objective.sense.maximize)

                cplex_instance.objective.set_name(
                    self._symbol_map.getSymbol(obj_data,
                                               self._labeler))

                if gen_obj_canonical_repn:
                    obj_repn = generate_canonical_repn(obj_data.expr)
                    block_canonical_repn[obj_data] = obj_repn
                else:
                    obj_repn = block_canonical_repn[obj_data]

                if (isinstance(obj_repn, LinearCanonicalRepn) and \
                    (obj_repn.linear == None)) or \
                    canonical_is_constant(obj_repn):
                    print("Warning: Constant objective detected, replacing "
                          "with a placeholder to prevent solver failure.")
                    offset = obj_repn.constant
                    if offset is None:
                        offset = 0.0
                    objective_expression = [("ONE_VAR_CONSTANT",offset)]
                    cplex_instance.objective.set_linear(objective_expression)

                else:

                    if isinstance(obj_repn, LinearCanonicalRepn):
                        objective_expression, offset = \
                            self._encode_constraint_body_linear_specialized(
                                    obj_repn,
                                    self._labeler,
                                    use_variable_names=False,
                                    cplex_variable_name_index_map=self._cplex_variable_ids,
                                    as_pairs=True)
                        if offset != 0.0:
                            objective_expression.append((self._cplex_variable_ids["ONE_VAR_CONSTANT"],offset))
                        cplex_instance.objective.set_linear(objective_expression)

                    else:
                        #Linear terms
                        if 1 in obj_repn:
                            objective_expression, offset = \
                                self._encode_constraint_body_linear(
                                    obj_repn,
                                    self._labeler,
                                    as_pairs=True)
                            if offset != 0.0:
                                objective_expression.append(("ONE_VAR_CONSTANT",offset))
                            cplex_instance.objective.set_linear(objective_expression)

                        #Quadratic terms
                        if 2 in obj_repn:
                            self._has_quadratic_objective = True
                            objective_expression = \
                                self._encode_constraint_body_quadratic(obj_repn,
                                                                       self._labeler,
                                                                       as_triples=True,
                                                                       is_obj=2.0)
                            cplex_instance.objective.\
                                set_quadratic_coefficients(objective_expression)

                        degree = canonical_degree(obj_repn)
                        if (degree is None) or (degree > 2):
                            raise ValueError(
                                "CPLEXPersistent plugin does not support general nonlinear "
                                "objective expressions (only linear or quadratic).\n"
                                "Objective: %s" % (obj_data.name))

    #
    # method to populate the CPLEX problem instance (interface) from
    # the supplied Pyomo problem instance.
    #
    def compile_instance(self,
                         pyomo_instance,
                         symbolic_solver_labels=False,
                         output_fixed_variable_bounds=False,
                         skip_trivial_constraints=False):

        from pyomo.core.base import Var, Constraint, SOSConstraint
        from pyomo.repn import canonical_is_constant, LinearCanonicalRepn, canonical_degree

        self._symbolic_solver_labels = symbolic_solver_labels
        self._output_fixed_variable_bounds = output_fixed_variable_bounds
        self._skip_trivial_constraints = skip_trivial_constraints

        self._has_quadratic_constraints = False
        self._has_quadratic_objective = False
        used_sos_constraints = False

        self._active_cplex_instance = cplex.Cplex()

        if self._symbolic_solver_labels:
            labeler = self._labeler = TextLabeler()
        else:
            labeler = self._labeler = NumericLabeler('x')

        self._symbol_map = SymbolMap()
        self._instance = pyomo_instance
        pyomo_instance.solutions.add_symbol_map(self._symbol_map)
        self._smap_id = id(self._symbol_map)

        # we use this when iterating over the constraints because it
        # will have a much smaller hash table, we also use this for
        # the warm start code after it is cleaned to only contain
        # variables referenced in the constraints
        self._variable_symbol_map = SymbolMap()

        # cplex wants the caller to set the problem type, which is (for
        # current purposes) strictly based on variable type counts.
        num_binary_variables = 0
        num_integer_variables = 0
        num_continuous_variables = 0

        #############################################
        # populate the variables in the cplex model #
        #############################################

        var_names = []
        var_lbs = []
        var_ubs = []
        var_types = []

        self._referenced_variable_ids.clear()

        # maps pyomo var data labels to the corresponding CPLEX variable id.
        self._cplex_variable_ids.clear()

        # cached in the loop below - used to update the symbol map
        # immediately following loop termination.
        var_label_pairs = []

        for var_data in pyomo_instance.component_data_objects(Var, active=True):

            if var_data.fixed and not self._output_fixed_variable_bounds:
                # if a variable is fixed, and we're preprocessing
                # fixed variables (as in not outputting them), there
                # is no need to add them to the compiled model.
                continue

            var_name = self._symbol_map.getSymbol(var_data, labeler)
            var_names.append(var_name)
            var_label_pairs.append((var_data, var_name))

            self._cplex_variable_ids[var_name] = len(self._cplex_variable_ids)

            if (var_data.lb is None) or (var_data.lb == -infinity):
                var_lbs.append(-cplex.infinity)
            else:
                var_lbs.append(value(var_data.lb))

            if (var_data.ub is None) or (var_data.ub == infinity):
                var_ubs.append(cplex.infinity)
            else:
                var_ubs.append(value(var_data.ub))

            if var_data.is_integer():
                var_types.append(self._active_cplex_instance.variables.type.integer)
                num_integer_variables += 1
            elif var_data.is_binary():
                var_types.append(self._active_cplex_instance.variables.type.binary)
                num_binary_variables += 1
            elif var_data.is_continuous():
                var_types.append(self._active_cplex_instance.variables.type.continuous)
                num_continuous_variables += 1
            else:
                raise TypeError("Invalid domain type for variable with name '%s'. "
                                "Variable is not continuous, integer, or binary.")

        self._active_cplex_instance.variables.add(names=var_names,
                                                  lb=var_lbs,
                                                  ub=var_ubs,
                                                  types=var_types)

        self._active_cplex_instance.variables.add(lb=[1],
                                                  ub=[1],
                                                  names=["ONE_VAR_CONSTANT"])

        self._cplex_variable_ids["ONE_VAR_CONSTANT"] = len(self._cplex_variable_ids)

        self._variable_symbol_map.addSymbols(var_label_pairs)
        self._cplex_variable_names = self._active_cplex_instance.variables.get_names()

        ########################################################
        # populate the standard constraints in the cplex model #
        ########################################################

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

        for block in pyomo_instance.block_data_objects(active=True):

            gen_con_canonical_repn = \
                getattr(block, "_gen_con_canonical_repn", True)
            # Get/Create the ComponentMap for the repn
            if not hasattr(block,'_canonical_repn'):
                block._canonical_repn = ComponentMap()
            block_canonical_repn = block._canonical_repn

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

                name = self._symbol_map.getSymbol(con, labeler)
                expr = None
                qexpr = None
                quadratic = False
                if isinstance(con_repn, LinearCanonicalRepn):
                    expr, offset = \
                        self._encode_constraint_body_linear_specialized(con_repn,
                                                                        labeler,
                                                                        use_variable_names=False,
                                                                        cplex_variable_name_index_map=self._cplex_variable_ids)
                else:
                    degree = canonical_degree(con_repn)
                    if degree == 2:
                        quadratic = True
                    elif (degree != 0) or (degree != 1):
                        raise ValueError(
                            "CPLEXPersistent plugin does not support general nonlinear "
                            "constraint expression (only linear or quadratic).\n"
                            "Constraint: %s" % (con.name))
                    expr, offset = self._encode_constraint_body_linear(con_repn,
                                                                       labeler)

                if quadratic:
                    if expr is None:
                        expr = cplex.SparsePair(ind=[0],val=[0.0])
                    self._has_quadratic_constraints = True

                    qexpr = self._encode_constraint_body_quadratic(con_repn,labeler)
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

        ###################################################
        # populate the SOS constraints in the cplex model #
        ###################################################

        # SOS constraints - largely taken from cpxlp.py so updates there,
        # should be applied here
        # TODO: Allow users to specify the variables coefficients for custom
        # branching/set orders - refer to cpxlp.py
        sosn = self._capabilities.sosn
        sos1 = self._capabilities.sos1
        sos2 = self._capabilities.sos2
        modelSOS = ModelSOS()
        for soscondata in pyomo_instance.component_data_objects(SOSConstraint,
                                                                active=True):
            level = soscondata.level
            if (level == 1 and not sos1) or \
               (level == 2 and not sos2) or \
               (level > 2 and not sosn):
                raise Exception("Solver does not support SOS level %s constraints"
                                % (level,))
            modelSOS.count_constraint(self._symbol_map,
                                      labeler,
                                      self._variable_symbol_map,
                                      soscondata)

        if modelSOS.sosType:
            for key in modelSOS.sosType:
                self._active_cplex_instance.SOS.add(type = modelSOS.sosType[key],
                                       name = modelSOS.sosName[key],
                                       SOS = [modelSOS.varnames[key],
                                              modelSOS.weights[key]])
                self._referenced_variable_ids.update(modelSOS.varids[key])
            used_sos_constraints = True

        self._active_cplex_instance.linear_constraints.add(
            lin_expr=expressions,
            senses=senses,
            rhs=rhss,
            range_values=range_values,
            names=names)

        for index in xrange(len(qexpressions)):
            self._active_cplex_instance.quadratic_constraints.add(
                lin_expr=qlinears[index],
                quad_expr=qexpressions[index],
                sense=qsenses[index],
                rhs=qrhss[index],
                name=qnames[index])

        #############################################
        # populate the objective in the cplex model #
        #############################################

        self.compile_objective(pyomo_instance)

        ################################################
        # populate the problem type in the cplex model #
        ################################################

        # This gets rid of the annoying "Freeing MIP data." message.
        def _filter_freeing_mip_data(val):
            if val.strip() == 'Freeing MIP data.':
                return ""
            return val
        self._active_cplex_instance.set_warning_stream(sys.stderr,
                                                       fn=_filter_freeing_mip_data)

        if (self._has_quadratic_objective is True) or \
           (self._has_quadratic_constraints is True):
            if (num_integer_variables > 0) or \
               (num_binary_variables > 0) or \
               (used_sos_constraints):
                if self._has_quadratic_constraints is True:
                    self._active_cplex_instance.set_problem_type(
                        self._active_cplex_instance.problem_type.MIQCP)
                else:
                    self._active_cplex_instance.set_problem_type(
                        self._active_cplex_instance.problem_type.MIQP)
            else:
                if self._has_quadratic_constraints is True:
                    self._active_cplex_instance.set_problem_type(
                        self._active_cplex_instance.problem_type.QCP)
                else:
                    self._active_cplex_instance.set_problem_type(
                        self._active_cplex_instance.problem_type.QP)
        elif (num_integer_variables > 0) or \
             (num_binary_variables > 0) or \
             (used_sos_constraints):
            self._active_cplex_instance.set_problem_type(
                self._active_cplex_instance.problem_type.MILP)
        else:
            self._active_cplex_instance.set_problem_type(
                self._active_cplex_instance.problem_type.LP)

        # restore the warning stream without our filter function
        self._active_cplex_instance.set_warning_stream(sys.stderr)


    #
    # simple method to query whether a Pyomo instance has already been
    # compiled.
    #
    def instance_compiled(self):

        return self._active_cplex_instance is not None

    #
    # Override base class method to check for compiled instance
    #
    def _warm_start(self, instance):

        if self._active_cplex_instance is None:
            raise RuntimeError("***The CPLEXPersistent solver plugin "
                               "cannot warm start - no instance is "
                               "presently compiled")

        # clear any existing warm starts.
        self._active_cplex_instance.MIP_starts.delete()

        # the iteration order is identical to that used in generating
        # the cplex instance, so all should be well.
        variable_ids = []
        variable_values = []

        # IMPT: the var_data returned is a weak ref!
        for label, var_data in iteritems(self._variable_symbol_map.bySymbol):
            cplex_id = self._cplex_variable_ids[label]
            if var_data().fixed and not self._output_fixed_variable_bounds:
                continue
            elif var_data().value is not None:
                variable_ids.append(cplex_id)
                variable_values.append(var_data().value)

        if len(variable_ids):
            self._active_cplex_instance.MIP_starts.add(
                [variable_ids, variable_values],
                self._active_cplex_instance.MIP_starts.effort_level.auto)

    #
    # Override base class method to check for compiled instance
    #

    def _populate_cplex_instance(self, model):
        assert model == self._instance

    def _presolve(self, *args, **kwds):

        if self._active_cplex_instance is None:
            raise RuntimeError("***The CPLEXPersistent solver plugin"
                               " cannot presolve - no instance is "
                               "presently compiled")

        # These must be passed in to the compile_instance method,
        # but assert that any values here match those already supplied
        if 'symbolic_solver_labels' in kwds:
            assert self._symbolic_solver_labels == \
                kwds['symbolic_solver_labels']
        if 'output_fixed_variable_bounds' in kwds:
            assert self._output_fixed_variable_bounds == \
                kwds['output_fixed_variable_bounds']
        if 'skip_trivial_constraints' in kwds:
            assert self._skip_trivial_constraints == \
                kwds["skip_trivial_constraints"]

        if self._smap_id not in self._instance.solutions.symbol_map:
            self._instance.solutions.add_symbol_map(self._symbol_map)

        CPLEXDirect._presolve(self, *args, **kwds)

        # like other solver plugins, persistent solver plugins can
        # take an instance as an input argument. the only context in
        # which this instance is used, however, is for warm-starting.
        if len(args) > 2:
            raise ValueError("The CPLEXPersistent plugin method "
                             "'_presolve' can be supplied at most "
                             "one problem instance - %s were "
                             "supplied" % len(args))

            # Re-add the symbol map id if it was cleared
            # after a previous solution load
            if id(self._symbol_map) not in args[0].solutions.symbol_map:
                args[0].solutions.add_symbol_map(self._symbol_map)
                self._smap_id = id(self._symbol_map)

    #
    # invoke the solver on the currently compiled instance!!!
    #
    def _apply_solver(self):

        if self._active_cplex_instance is None:
            raise RuntimeError("***The CPLEXPersistent solver plugin cannot "
                               "apply solver - no instance is presently compiled")

        # NOTE:
        # CPLEX maintains the pool of feasible solutions from the
        # prior solve as the set of mip starts for the next solve.
        # and evaluating multiple mip starts (and there can be many)
        # is expensive. so if the warm_start method is not invoked,
        # there will potentially be a lot of time wasted.

        return CPLEXDirect._apply_solver(self)

    def _postsolve(self):

        if self._active_cplex_instance is None:
            raise RuntimeError("***The CPLEXPersistent solver plugin "
                               "cannot postsolve - no instance is "
                               "presently compiled")

        active_cplex_instance = self._active_cplex_instance
        variable_symbol_map = self._variable_symbol_map
        instance = self._instance

        ret = CPLEXDirect._postsolve(self)

        #
        # These get reset to None by the base class method
        #
        self._active_cplex_instance = active_cplex_instance
        self._variable_symbol_map = variable_symbol_map
        self._instance = instance

        return ret

if cplex_import_available is False:
    SolverFactory().deactivate('_cplex_persistent')
    SolverFactory().deactivate('_mock_cplexpersistent')
