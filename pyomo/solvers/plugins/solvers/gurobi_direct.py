#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
import re
import itertools
import math
import sys

from six import itervalues, iteritems
from six.moves import xrange

logger = logging.getLogger('pyomo.solvers')

_GUROBI_VERSION_MAJOR = None
_gurobi_version = None
gurobi_python_api_exists = None
def configure_gurobi_direct():
    global _GUROBI_VERSION_MAJOR
    global _gurobi_version
    global gurobi_python_api_exists
    if _gurobi_version is not None:
        return
    try:
        # import all the glp_* functions
        import gurobipy
        gurobi_direct._gurobi_module = gurobipy
        # create a version tuple of length 4
        _gurobi_version = gurobipy.gurobi.version()
        while(len(_gurobi_version) < 4):
            _gurobi_version += (0,)
        _gurobi_version = _gurobi_version[:4]
        _GUROBI_VERSION_MAJOR = _gurobi_version[0]
        gurobi_python_api_exists = True
    except ImportError:
        gurobi_python_api_exists = False
    except Exception as e:
        # other forms of exceptions can be thrown by the gurobi python
        # import. for example, a gurobipy.GurobiError exception is thrown
        # if all tokens for Gurobi are already in use. assuming, of
        # course, the license is a token license. unfortunately, you can't
        # import without a license, which means we can't test for the
        # exception above!
        print("Import of gurobipy failed - gurobi message="+str(e)+"\n")
        gurobi_python_api_exists = False

import pyutilib.services
from pyutilib.misc import Bunch, Options

from pyomo.util.plugin import alias
from pyomo.opt.base import *
from pyomo.opt.base.solvers import _extract_version
from pyomo.opt.results import *
from pyomo.opt.solver import *
from pyomo.core.base import (SymbolMap,
                             ComponentMap,
                             NumericLabeler,
                             is_fixed,
                             TextLabeler)
from pyomo.repn import generate_canonical_repn, LinearCanonicalRepn, canonical_degree

from pyomo.core.kernel.component_block import IBlockStorage
from pyomo.solvers.plugins.solvers.direct_solver import DirectSolver
from pyomo.core.kernel.numvalue import value
import pyomo.core.kernel
from pyomo.opt.results.problem import ProblemSense

GRB_MAX = -1
GRB_MIN = 1


class ModelSOS(object):

    def __init__(self):
        self.sosType = {}
        self.sosName = {}
        self.varnames = {}
        self.varids = {}
        self.weights = {}
        self.block_cntr = 0

    def count_constraint(self,symbol_map,labeler,variable_symbol_map,gurobi_var_map,soscondata):

        if hasattr(soscondata, 'get_items'):
            sos_items = list(soscondata.get_items())
        else:
            sos_items = list(soscondata.items())
        level = soscondata.level

        if len(sos_items) == 0:
            return

        self.block_cntr += 1
        varnames = self.varnames[self.block_cntr] = []
        varids = self.varids[self.block_cntr] = []
        weights = self.weights[self.block_cntr] = []
        if level == 1:
            self.sosType[self.block_cntr] = gurobi_direct._gurobi_module.GRB.SOS_TYPE1
        elif level == 2:
            self.sosType[self.block_cntr] = gurobi_direct._gurobi_module.GRB.SOS_TYPE2
        else:
            raise ValueError("Unsupported SOSConstraint level %s" % level)

        self.sosName[self.block_cntr] = symbol_map.getSymbol(soscondata,labeler)

        for vardata, weight in sos_items:
            if vardata.fixed:
                raise RuntimeError("SOSConstraint '%s' includes a fixed variable "
                                   "'%s'. This is currently not supported. "
                                   "Deactivate this constraint in order to "
                                   "proceed." % (soscondata.name,
                                                 vardata.name))
            varids.append(id(vardata))
            varnames.append(gurobi_var_map[variable_symbol_map.getSymbol(vardata)])
            weights.append(weight)

class gurobi_direct ( OptSolver ):
    """The Gurobi optimization solver (direct API plugin)

 The gurobi_direct plugin offers an API interface to Gurobi.  It requires the
 Python Gurobi API interface (gurobipy) be in Pyomo's lib/ directory.  Generally, if you can run Pyomo's Python instance, and execute

 >>> import gurobipy
 >>>

 with no errors, then this plugin will be enabled.

 Because of the direct connection with the Gurobi, no temporary files need be
 written or read.  That ostensibly makes this a faster plugin than the file-based
 Gurobi plugin.  However, you will likely not notice any speed up unless you are
 using the GLPK solver with PySP problems (due to the rapid re-solves).

 One downside to the lack of temporary files, is that there is no LP file to
 inspect for clues while debugging a model.  For that, use the 'write' solver
 option:

 $ pyomo model.{py,dat} \
   --solver=gurobi_direct \
   --solver-options  write=/path/to/some/file.lp

 This is a direct interface to Gurobi's Model.write function, the extension of the file is important.  You could, for example, write the file in MPS format:

 $ pyomo model.{py,dat} \
   --solver=gurobi_direct \
   --solver-options  write=/path/to/some/file.mps

    """

    alias('_gurobi_direct',
          doc='Direct Python interface to the Gurobi optimization solver.')

    _gurobi_module = None

    def __init__(self, **kwds):
        configure_gurobi_direct()
        #
        # Call base class constructor
        #
        kwds['type'] = 'gurobi_direct'
        OptSolver.__init__(self, **kwds)

        self._model = None

        # a dictionary that maps pyomo _VarData labels to the
        # corresponding Gurobi variable object. created each time
        # _populate_gurobi_instance is called.
        self._pyomo_gurobi_variable_map = None

        # this interface doesn't use files, but we can create a log
        # file is requested
        self._keepfiles = False
        # do we warmstart
        self._warm_start_solve = False
        # io_options
        self._symbolic_solver_labels = False
        self._output_fixed_variable_bounds = False
        self._skip_trivial_constraint = False

        # Note: Undefined capabilities default to 'None'
        self._capabilities = Options()
        self._capabilities.linear = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = True
        self._capabilities.integer = True
        self._capabilities.sos1 = True
        self._capabilities.sos2 = True

        # collection of id(_VarData).
        self._referenced_variable_ids = set()

    def _get_version(self):
        if _gurobi_version is None:
            return _extract_version('')
        return _gurobi_version

    def available(self, exception_flag=True):
        """ True if the solver is available """

        if exception_flag is False:
            return gurobi_python_api_exists
        else:
            if gurobi_python_api_exists is False:
                raise pyutilib.common.ApplicationError("No Gurobi <-> Python bindings available - Gurobi direct solver functionality is not available")
            else:
                return True

    def _get_bound(self, exp):
        if exp is None:
            return None
        if is_fixed(exp):
            return value(exp)
        raise ValueError("non-fixed bound: " + str(exp))

    def _populate_gurobi_instance (self, pyomo_instance):

        from pyomo.core.base import Var, Objective, Constraint, SOSConstraint
        from pyomo.repn import LinearCanonicalRepn, canonical_degree

        try:
            grbmodel = gurobi_direct._gurobi_module.Model()
        except Exception:
            e = sys.exc_info()[1]
            msg = 'Unable to create Gurobi model.  Have you installed the Python'\
            '\n       bindings for Gurobi?\n\n\tError message: %s'
            raise Exception(msg % e)

        if self._symbolic_solver_labels:
            labeler = TextLabeler()
        else:
            labeler = NumericLabeler('x')
        # cache to avoid dictionary getitem calls in the loops below.
        symbol_map = self._symbol_map = SymbolMap()
        self._smap_id = id(symbol_map)
        if isinstance(pyomo_instance, IBlockStorage):
            # BIG HACK (see pyomo.core.kernel write function)
            if not hasattr(pyomo_instance, "._symbol_maps"):
                setattr(pyomo_instance, "._symbol_maps", {})
            getattr(pyomo_instance,
                    "._symbol_maps")[self._smap_id] = symbol_map
        else:
            pyomo_instance.solutions.add_symbol_map(symbol_map)

        # we use this when iterating over the constraints because it
        # will have a much smaller hash table, we also use this for
        # the warm start code after it is cleaned to only contain
        # variables referenced in the constraints
        self_variable_symbol_map = self._variable_symbol_map = SymbolMap()
        var_symbol_pairs = []

        # maps _VarData labels to the corresponding Gurobi variable object
        pyomo_gurobi_variable_map = {}

        self._referenced_variable_ids.clear()

        # cache to avoid dictionary getitem calls in the loop below.
        grb_infinity = gurobi_direct._gurobi_module.GRB.INFINITY

        for var_value in pyomo_instance.component_data_objects(Var, active=True):

            lb = -grb_infinity
            ub = grb_infinity

            if var_value.has_lb():
                lb = value(var_value.lb)
            if var_value.has_ub():
                ub = value(var_value.ub)

            # _VarValue objects will not be in the symbol map yet, so
            # avoid some checks.
            var_value_label = symbol_map.createSymbol(var_value, labeler)
            var_symbol_pairs.append((var_value, var_value_label))

            # be sure to impart the integer and binary nature of any variables
            if var_value.is_binary():
                var_type = gurobi_direct._gurobi_module.GRB.BINARY
            elif var_value.is_integer():
                var_type = gurobi_direct._gurobi_module.GRB.INTEGER
            elif var_value.is_continuous():
                var_type = gurobi_direct._gurobi_module.GRB.CONTINUOUS
            else:
                raise TypeError("Invalid domain type for variable with name '%s'. "
                                "Variable is not continuous, integer, or binary.")

            pyomo_gurobi_variable_map[var_value_label] = \
                grbmodel.addVar(lb=lb, \
                                ub=ub, \
                                vtype=var_type, \
                                name=var_value_label)

        self_variable_symbol_map.addSymbols(var_symbol_pairs)

        grbmodel.update()

        # The next loop collects the following component types from the model:
        #  - SOSConstraint
        #  - Objective
        #  - Constraint
        sos1 = self._capabilities.sos1
        sos2 = self._capabilities.sos2
        modelSOS = ModelSOS()
        objective_cntr = 0
        # Track the range constraints and their associated variables added by gurobi
        self._last_native_var_idx = grbmodel.NumVars-1
        range_var_idx = grbmodel.NumVars
        _self_range_con_var_pairs = self._range_con_var_pairs = []
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
                        "Solver does not support SOS level %s constraints" % (level,))
                modelSOS.count_constraint(symbol_map,
                                          labeler,
                                          self_variable_symbol_map,
                                          pyomo_gurobi_variable_map,
                                          soscondata)

            # Objective
            for obj_data in block.component_data_objects(Objective,
                                                         active=True,
                                                         descend_into=False):

                if objective_cntr > 1:
                    raise ValueError(
                        "Multiple active objectives found on Pyomo instance '%s'. "
                        "Solver '%s' will only handle a single active objective" \
                        % (pyomo_instance.name, self.type))

                sense = GRB_MIN if (obj_data.is_minimizing()) else GRB_MAX
                grbmodel.ModelSense = sense
                obj_expr = gurobi_direct._gurobi_module.LinExpr()

                if gen_obj_canonical_repn:
                    obj_repn = generate_canonical_repn(obj_data.expr)
                    block_canonical_repn[obj_data] = obj_repn
                else:
                    obj_repn = block_canonical_repn[obj_data]

                if isinstance(obj_repn, LinearCanonicalRepn):

                    if obj_repn.constant is not None:
                        obj_expr.addConstant(obj_repn.constant)

                    if (obj_repn.linear is not None) and \
                       (len(obj_repn.linear) > 0):

                        for i in xrange(len(obj_repn.linear)):
                            var_coefficient = obj_repn.linear[i]
                            var_value = obj_repn.variables[i]
                            self._referenced_variable_ids.add(id(var_value))
                            label = self_variable_symbol_map.getSymbol(var_value)
                            obj_expr.addTerms(var_coefficient,
                                              pyomo_gurobi_variable_map[label])
                else:

                    if 0 in obj_repn: # constant term
                        obj_expr.addConstant(obj_repn[0][None])

                    if 1 in obj_repn: # first-order terms
                        hash_to_variable_map = obj_repn[-1]
                        for var_hash, var_coefficient in iteritems(obj_repn[1]):
                            vardata = hash_to_variable_map[var_hash]
                            self._referenced_variable_ids.add(id(vardata))
                            label = self_variable_symbol_map.getSymbol(vardata)
                            obj_expr.addTerms(var_coefficient,
                                              pyomo_gurobi_variable_map[label])

                    if 2 in obj_repn:
                        obj_expr = gurobi_direct._gurobi_module.QuadExpr(obj_expr)
                        hash_to_variable_map = obj_repn[-1]
                        for quad_repn, coef in iteritems(obj_repn[2]):
                            gurobi_expr = gurobi_direct._gurobi_module.QuadExpr(coef)
                            for var_hash, exponent in iteritems(quad_repn):
                                vardata = hash_to_variable_map[var_hash]
                                self._referenced_variable_ids.add(id(vardata))
                                gurobi_var = pyomo_gurobi_variable_map\
                                             [self_variable_symbol_map.\
                                              getSymbol(vardata)]
                                gurobi_expr *= gurobi_var
                                if exponent == 2:
                                    gurobi_expr *= gurobi_var
                            obj_expr += gurobi_expr

                    degree = canonical_degree(obj_repn)
                    if (degree is None) or (degree > 2):
                        raise ValueError(
                            "gurobi_direct plugin does not support general nonlinear "
                            "objective expressions (only linear or quadratic).\n"
                            "Objective: %s" % (obj_data.name))

                # need to cache the objective label, because the
                # GUROBI python interface doesn't track this.
                # _ObjectiveData objects will not be in the symbol map
                # yet, so avoid some checks.
                self._objective_label = \
                    symbol_map.createSymbol(obj_data, labeler)

                grbmodel.setObjective(obj_expr, sense=sense)

            # Constraint
            for constraint_data in block.component_data_objects(Constraint,
                                                                active=True,
                                                                descend_into=False):

                if (not constraint_data.has_lb()) and \
                   (not constraint_data.has_ub()):
                    assert not constraint_data.equality
                    continue  # not binding at all, don't bother

                con_repn = None
                if constraint_data._linear_canonical_form:
                    con_repn = constraint_data.canonical_form()
                elif isinstance(constraint_data, LinearCanonicalRepn):
                    con_repn = constraint_data
                else:
                    if gen_con_canonical_repn:
                        con_repn = generate_canonical_repn(constraint_data.body)
                        block_canonical_repn[constraint_data] = con_repn
                    else:
                        con_repn = block_canonical_repn[constraint_data]

                offset = 0.0
                # _ConstraintData objects will not be in the symbol
                # map yet, so avoid some checks.
                constraint_label = \
                    symbol_map.createSymbol(constraint_data, labeler)

                trivial = False
                if isinstance(con_repn, LinearCanonicalRepn):

                    #
                    # optimization (these might be generated on the fly)
                    #
                    constant = con_repn.constant
                    coefficients = con_repn.linear
                    variables = con_repn.variables

                    if constant is not None:
                        offset = constant
                    expr = gurobi_direct._gurobi_module.LinExpr() + offset

                    if (coefficients is not None) and \
                       (len(coefficients) > 0):

                        linear_coefs = list()
                        linear_vars = list()

                        for i in xrange(len(coefficients)):

                            var_coefficient = coefficients[i]
                            var_value = variables[i]
                            self._referenced_variable_ids.add(id(var_value))
                            label = self_variable_symbol_map.getSymbol(var_value)
                            linear_coefs.append(var_coefficient)
                            linear_vars.append(pyomo_gurobi_variable_map[label])

                        expr += gurobi_direct._gurobi_module.LinExpr(linear_coefs, linear_vars)

                    else:

                        trivial = True

                else:

                    if 0 in con_repn:
                        offset = con_repn[0][None]
                    expr = gurobi_direct._gurobi_module.LinExpr() + offset

                    if 1 in con_repn: # first-order terms

                        linear_coefs = list()
                        linear_vars = list()

                        hash_to_variable_map = con_repn[-1]
                        for var_hash, var_coefficient in iteritems(con_repn[1]):
                            var = hash_to_variable_map[var_hash]
                            self._referenced_variable_ids.add(id(var))
                            label = self_variable_symbol_map.getSymbol(var)
                            linear_coefs.append( var_coefficient )
                            linear_vars.append( pyomo_gurobi_variable_map[label] )

                        expr += gurobi_direct._gurobi_module.LinExpr(linear_coefs, linear_vars)

                    if 2 in con_repn: # quadratic constraint
                        if _GUROBI_VERSION_MAJOR < 5:
                            raise ValueError(
                                "The gurobi_direct plugin does not handle quadratic "
                                "constraint expressions for Gurobi major versions "
                                "< 5. Current version: Gurobi %s.%s%s"
                                % (gurobi_direct._gurobi_module.gurobi.version()))

                        expr = gurobi_direct._gurobi_module.QuadExpr(expr)
                        hash_to_variable_map = con_repn[-1]
                        for quad_repn, coef in iteritems(con_repn[2]):
                            gurobi_expr = gurobi_direct._gurobi_module.QuadExpr(coef)
                            for var_hash, exponent in iteritems(quad_repn):
                                vardata = hash_to_variable_map[var_hash]
                                self._referenced_variable_ids.add(id(vardata))
                                gurobi_var = pyomo_gurobi_variable_map\
                                             [self_variable_symbol_map.\
                                              getSymbol(vardata)]
                                gurobi_expr *= gurobi_var
                                if exponent == 2:
                                    gurobi_expr *= gurobi_var
                            expr += gurobi_expr

                    degree = canonical_degree(con_repn)
                    if (degree is None) or (degree > 2):
                        raise ValueError(
                            "gurobi_direct plugin does not support general nonlinear "
                            "constraint expressions (only linear or quadratic).\n"
                            "Constraint: %s" % (constraint_data.name))

                if (not trivial) or (not self._skip_trivial_constraints):

                    if constraint_data.equality:
                        sense = gurobi_direct._gurobi_module.GRB.EQUAL
                        bound = self._get_bound(constraint_data.lower)
                        grbmodel.addConstr(lhs=expr,
                                           sense=sense,
                                           rhs=bound,
                                           name=constraint_label)
                    else:
                        # L <= body <= U
                        if constraint_data.has_lb() and \
                           constraint_data.has_ub():
                            grb_con = grbmodel.addRange(
                                expr,
                                self._get_bound(constraint_data.lower),
                                self._get_bound(constraint_data.upper),
                                constraint_label)
                            _self_range_con_var_pairs.append((grb_con,range_var_idx))
                            range_var_idx += 1
                        # body <= U
                        elif constraint_data.has_ub():
                            bound = self._get_bound(constraint_data.upper)
                            if bound < float('inf'):
                                grbmodel.addConstr(
                                    lhs=expr,
                                    sense=gurobi_direct._gurobi_module.GRB.LESS_EQUAL,
                                    rhs=bound,
                                    name=constraint_label
                                    )
                        # L <= body
                        else:
                            assert constraint_data.has_lb()
                            bound = self._get_bound(constraint_data.lower)
                            if bound > -float('inf'):
                                grbmodel.addConstr(
                                    lhs=expr,
                                    sense=gurobi_direct._gurobi_module.GRB.GREATER_EQUAL,
                                    rhs=bound,
                                    name=constraint_label
                                    )

        if modelSOS.sosType:
            for key in modelSOS.sosType:
                grbmodel.addSOS(modelSOS.sosType[key], \
                                modelSOS.varnames[key], \
                                modelSOS.weights[key] )
                self._referenced_variable_ids.update(modelSOS.varids[key])

        for var_id in self._referenced_variable_ids:
            varname = self._variable_symbol_map.byObject[var_id]
            vardata = self._variable_symbol_map.bySymbol[varname]()
            if vardata.fixed:
                if not self._output_fixed_variable_bounds:
                    raise ValueError("Encountered a fixed variable (%s) inside an active objective "
                                     "or constraint expression on model %s, which is usually indicative of "
                                     "a preprocessing error. Use the IO-option 'output_fixed_variable_bounds=True' "
                                     "to suppress this error and fix the variable by overwriting its bounds in "
                                     "the Gurobi instance."
                                     % (vardata.name,pyomo_instance.name,))

                grbvar = pyomo_gurobi_variable_map[varname]
                grbvar.setAttr(gurobi_direct._gurobi_module.GRB.Attr.UB, vardata.value)
                grbvar.setAttr(gurobi_direct._gurobi_module.GRB.Attr.LB, vardata.value)

        grbmodel.update()

        self._gurobi_instance = grbmodel
        self._pyomo_gurobi_variable_map = pyomo_gurobi_variable_map

    def warm_start_capable(self):

        return True

    def _warm_start(self, instance):

        for symbol, vardata_ref in iteritems(self._variable_symbol_map.bySymbol):
            vardata = vardata_ref()
            if vardata.value is not None:
                self._pyomo_gurobi_variable_map[symbol].setAttr(
                    gurobi_direct._gurobi_module.GRB.Attr.Start,
                    vardata.value)

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
                             create_tempfile(suffix = '.gurobi.log')

        # Possible TODOs
        if self._timelimit is not None:
            logger.warning("The 'timelimit' keyword will be ignored "
                           "for solver="+self.type)
        if self._soln_file is not None:
            logger.warning("The 'soln_file' keyword will be ignored "
                           "for solver="+self.type)

        model = args[0]
        if len(args) != 1:
            msg = "The gurobi_direct plugin method '_presolve' must be supplied "\
                  "a single problem instance - %s were supplied"
            raise ValueError(msg % len(args))
        elif not isinstance(model, (Model, IBlockStorage)):
            raise ValueError("The problem instance supplied to the "            \
                 "gurobi_direct plugin '_presolve' method must be of type 'Model'")

        self._populate_gurobi_instance(model)
        grbmodel = self._gurobi_instance

        # Clean up the symbol map to only contain variables referenced
        # in the constraints
        # **NOTE**: The warmstart method (if called below),
        #           relies on a "clean" symbol map
        vars_to_delete = set(self._variable_symbol_map.byObject.keys()) - \
                         self._referenced_variable_ids
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
            grbmodel.write( fname )

        if self._warm_start_solve:

            if len(args) != 1:
                msg = "The gurobi_direct _presolve method can only handle a single"\
                      "problem instance - %s were supplied"
                raise ValueError(msg % len(args))

            self._warm_start(model)

        # This does not use weak references so we need to release
        # references to external model variables
        del self._variable_symbol_map


    def _apply_solver(self):
        # TODO apply appropriate user-specified parameters

        prob = self._gurobi_instance

        if self._tee:
            prob.setParam('OutputFlag', 1)
        else:
            prob.setParam('OutputFlag', 0)

        prob.setParam('LogFile', self._log_file)

        if self._keepfiles:
            print("Solver log file: "+self._log_file)

        #Options accepted by gurobi (case insensitive):
        #['Cutoff', 'IterationLimit', 'NodeLimit', 'SolutionLimit', 'TimeLimit',
        # 'FeasibilityTol', 'IntFeasTol', 'MarkowitzTol', 'MIPGap', 'MIPGapAbs',
        # 'OptimalityTol', 'PSDTol', 'Method', 'PerturbValue', 'ObjScale', 'ScaleFlag',
        # 'SimplexPricing', 'Quad', 'NormAdjust', 'BarIterLimit', 'BarConvTol',
        # 'BarCorrectors', 'BarOrder', 'Crossover', 'CrossoverBasis', 'BranchDir',
        # 'Heuristics', 'MinRelNodes', 'MIPFocus', 'NodefileStart', 'NodefileDir',
        # 'NodeMethod', 'PumpPasses', 'RINS', 'SolutionNumber', 'SubMIPNodes', 'Symmetry',
        # 'VarBranch', 'Cuts', 'CutPasses', 'CliqueCuts', 'CoverCuts', 'CutAggPasses',
        # 'FlowCoverCuts', 'FlowPathCuts', 'GomoryPasses', 'GUBCoverCuts', 'ImpliedCuts',
        # 'MIPSepCuts', 'MIRCuts', 'NetworkCuts', 'SubMIPCuts', 'ZeroHalfCuts', 'ModKCuts',
        # 'Aggregate', 'AggFill', 'PreDual', 'DisplayInterval', 'IISMethod', 'InfUnbdInfo',
        # 'LogFile', 'PreCrush', 'PreDepRow', 'PreMIQPMethod', 'PrePasses', 'Presolve',
        # 'ResultFile', 'ImproveStartTime', 'ImproveStartGap', 'Threads', 'Dummy', 'OutputFlag']
        for key in self.options:
            prob.setParam( key, self.options[key] )

        if 'relax_integrality' in self.options:
            for v in prob.getVars():
                if v.vType != gurobi_direct._gurobi_module.GRB.CONTINUOUS:
                    v.vType = gurobi_direct._gurobi_module.GRB.CONTINUOUS
            prob.update()

        if _GUROBI_VERSION_MAJOR >= 5:
            for suffix in self._suffixes:
                if re.match(suffix, "dual"):
                    prob.setParam(
                        gurobi_direct._gurobi_module.GRB.Param.QCPDual,
                        1)

        # Actually solve the problem.
        prob.optimize()

        prob.setParam('LogFile', 'default')

        # FIXME: can we get a return code indicating if Gurobi had a
        # significant failure?
        return Bunch(rc=None, log=None)

    def _gurobi_get_solution_status ( self ):
        status = self._gurobi_instance.Status
        if   gurobi_direct._gurobi_module.GRB.OPTIMAL == status:
            return SolutionStatus.optimal
        elif gurobi_direct._gurobi_module.GRB.INFEASIBLE == status:
            return SolutionStatus.infeasible
        elif gurobi_direct._gurobi_module.GRB.CUTOFF == status:
            return SolutionStatus.other
        elif gurobi_direct._gurobi_module.GRB.INF_OR_UNBD == status:
            return SolutionStatus.other
        elif gurobi_direct._gurobi_module.GRB.INTERRUPTED == status:
            return SolutionStatus.other
        elif gurobi_direct._gurobi_module.GRB.LOADED == status:
            return SolutionStatus.other
        elif gurobi_direct._gurobi_module.GRB.SUBOPTIMAL == status:
            return SolutionStatus.other
        elif gurobi_direct._gurobi_module.GRB.UNBOUNDED == status:
            return SolutionStatus.other
        elif gurobi_direct._gurobi_module.GRB.ITERATION_LIMIT == status:
            return SolutionStatus.stoppedByLimit
        elif gurobi_direct._gurobi_module.GRB.NODE_LIMIT == status:
            return SolutionStatus.stoppedByLimit
        elif gurobi_direct._gurobi_module.GRB.SOLUTION_LIMIT == status:
            return SolutionStatus.stoppedByLimit
        elif gurobi_direct._gurobi_module.GRB.TIME_LIMIT == status:
            return SolutionStatus.stoppedByLimit
        elif gurobi_direct._gurobi_module.GRB.NUMERIC == status:
            return SolutionStatus.error
        raise RuntimeError("Unknown solution status returned by "
                           "Gurobi solver")

    def _postsolve(self):

        # the only suffixes that we extract from GUROBI are
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
                    "***The gurobi_direct solver plugin "
                    "cannot extract solution suffix="+suffix)

        gprob = self._gurobi_instance

        if (gprob.getAttr(gurobi_direct._gurobi_module.GRB.Attr.IsMIP)):
            extract_reduced_costs = False
            extract_duals = False

        pvars = gprob.getVars()
        cons = gprob.getConstrs()
        qcons = []
        if _GUROBI_VERSION_MAJOR >= 5:
            qcons = gprob.getQConstrs()

        results = SolverResults()
        soln = Solution()
        problem = results.problem
        solver  = results.solver

        # cache the variable and constraint dictionaries -
        # otherwise, each invocation will include a lookup in a
        # MapContainer, which is extremely expensive.
        soln_variables = soln.variable
        soln_constraints = soln.constraint

        solver.name = ("Gurobi %s.%s%s"
                       % gurobi_direct._gurobi_module.gurobi.version())
        solver.wallclock_time = gprob.Runtime

        if gprob.Status == 1: # problem is loaded, but no solution
            solver.termination_message = "Model is loaded, but no solution information is available."
            solver.termination_condition = TerminationCondition.other
        if gprob.Status == 2: # optimal
            solver.termination_message = "Model was solved to optimality (subject to tolerances)."
            solver.termination_condition = TerminationCondition.optimal
        elif gprob.Status == 3: # infeasible
            solver.termination_message = "Model was proven to be infeasible."
            solver.termination_condition = TerminationCondition.infeasible
        elif gprob.Status == 4: # infeasible or unbounded
            solver.termination_message = "Model was proven to be either infeasible or unbounded."
            solver.termination_condition = TerminationCondition.infeasibleOrUnbounded
        elif gprob.Status == 5: # unbounded
            solver.termination_message = "Model was proven to be unbounded."
            solver.termination_condition = TerminationCondition.unbounded
        elif gprob.Status == 6: # cutoff
            solver.termination_message = "Optimal objective for model was proven to be worse than the value specified in the Cutoff parameter."
            solver.termination_condition = TerminationCondition.minFunctionValue
        elif gprob.Status == 7: # iteration limit
            solver.termination_message = "Optimization terminated because the total number of simplex or barrier iterations exceeded specified limits."
            solver.termination_condition = TerminationCondition.maxIterations
        elif gprob.Status == 8: # node limit
            solver.termination_message = "Optimization terminated because the total number of branch-and-cut nodes exceeded specified limits."
            solver.termination_condition = TerminationCondition.maxEvaluations
        elif gprob.Status == 9: # time limit
            solver.termination_message = "Optimization terminated because the total time expended exceeded specified limits."
            solver.termination_condition = TerminationCondition.maxTimeLimit
        elif gprob.Status == 10: # solution limit
            solver.termination_message = "Optimization terminated because the number of solutions found reached specified limits."
            solver.termination_condition = TerminationCondition.other
        elif gprob.Status == 11: # interrupted
            solver.termination_message = "Optimization was terminated by the user."
            solver.termination_condition = TerminationCondition.other
        elif gprob.Status == 12: # numeric issues
            solver.termination_message = "Optimization was terminated due to unrecoverable numerical difficulties."
            solver.termination_condition = TerminationCondition.other
        elif gprob.Status == 13: # suboptimal
            solver.termination_message = "Optimization was unable to satisfy optimality tolerances; returned solution is sub-optimal."
            solver.termination_condition = TerminationCondition.other
        else: # unknown
            solver.termination_message = "Unknown termination condition received following optimization."
            solver.termination_condition = TerminationCondition.other

        problem.name = gprob.ModelName
        if gprob.ModelSense > 0: # positive numbers indicate minimization
            # depending on whether the problem is a MIP or an LP, and whether
            # or not the solution is feasible, one or two of the attributes
            # below may not exist, yielding an exception as a result of the query.
            try:
                problem.upper_bound = gprob.ObjVal
                problem.lower_bound = gprob.ObjBound
            except:
                pass
        else:
            try:
                problem.upper_bound = gprob.ObjBound
                problem.lower_bound = gprob.ObjVal
            except:
                pass

        problem.number_of_constraints          = len(cons)+len(qcons)+gprob.NumSOS
        problem.number_of_nonzeros             = gprob.NumNZs
        problem.number_of_variables            = gprob.NumVars
        problem.number_of_binary_variables     = gprob.NumBinVars
        problem.number_of_integer_variables    = gprob.NumIntVars
        problem.number_of_continuous_variables = gprob.NumVars \
                                                - gprob.NumIntVars \
                                                - gprob.NumBinVars
        problem.number_of_objectives = 1
        problem.number_of_solutions = gprob.SolCount

        problem.sense = ProblemSense.minimize
        if problem.sense == GRB_MAX:
            problem.sense = ProblemSense.maximize

        soln.status = self._gurobi_get_solution_status()
        soln.gap = None # until proven otherwise

        # if a solve was stopped by a limit, we still need to check to
        # see if there is a solution available - this may not always
        # be the case, both in LP and MIP contexts.
        if (soln.status == SolutionStatus.optimal) or \
           ((soln.status == SolutionStatus.stoppedByLimit) and \
            (gprob.SolCount > 0)):

            obj_val = gprob.ObjVal

            if (problem.number_of_binary_variables + \
                problem.number_of_integer_variables) > 0:
                obj_bound = gprob.ObjBound

            if problem.sense == ProblemSense.minimize:
                if (problem.number_of_binary_variables + \
                    problem.number_of_integer_variables) == 0:
                    problem.upper_bound = obj_val
                else:
                    problem.lower_bound = obj_bound
                    problem.upper_bound = obj_val
            else:

                if (problem.number_of_binary_variables + \
                    problem.number_of_integer_variables) == 0:
                    problem.lower_bound = obj_val
                else:
                    problem.upper_bound = obj_bound
                    problem.lower_bound = obj_val

            soln.objective[self._objective_label] = \
                {'Value': obj_val}
            if (problem.number_of_binary_variables + \
                problem.number_of_integer_variables) == 0:
                soln.gap = None
            else:
                soln.gap = math.fabs(obj_val - obj_bound)

            # Those variables not added by gurobi due to range constraints
            for var in itertools.islice(pvars, self._last_native_var_idx + 1):
                soln_variables[ var.VarName ] = {"Value" : var.X}

            if extract_reduced_costs:
                for var in itertools.islice(pvars, self._last_native_var_idx + 1):
                    soln_variables[ var.VarName ]["Rc"] = var.Rc

            if extract_duals or extract_slacks:
                for con in cons:
                    soln_constraints[con.ConstrName] = {}
                for con in qcons:
                    soln_constraints[con.QCName] = {}

            if extract_duals:
                for con in cons:
                    # Pi attributes in Gurobi are the constraint duals
                    soln_constraints[ con.ConstrName ]["Dual"] = con.Pi
                for con in qcons:
                    # QCPI attributes in Gurobi are the constraint duals
                    soln_constraints[ con.QCName ]["Dual"] = con.QCPi

            if extract_slacks:
                for con in cons:
                    soln_constraints[ con.ConstrName ]["Slack"] = con.Slack
                for con in qcons:
                    soln_constraints[ con.QCName ]["Slack"] = con.QCSlack
                # The above loops may include range constraints but will
                # always report a slack of zero since gurobi transforms
                # range constraints by adding a slack variable in the following way
                # L <= f(x) <= U
                # becomes
                # 0 <= U-f(x) <= U-L
                # becomes
                # U-f(x) == s
                # 0 <= s <= U-L
                # Therefore we need to check the value of the
                # associated slack variable with its upper bound to
                # compute the original constraint slacks. To conform
                # with the other problem writers we return the slack
                # value that is largest in magnitude (L-f(x) or
                # U-f(x))
                for con,var_idx in self._range_con_var_pairs:
                    var = pvars[var_idx]
                    # U-f(x)
                    Us_ = var.X
                    # f(x)-L
                    Ls_ = var.UB-var.X
                    if Us_ > Ls_:
                        soln_constraints[ con.ConstrName ]["Slack"] = Us_
                    else:
                        soln_constraints[ con.ConstrName ]["Slack"] = -Ls_

            byObject = self._symbol_map.byObject
            referenced_varnames = \
                set(byObject[varid]
                    for varid in self._referenced_variable_ids)
            names_to_delete = \
                set(soln_variables.keys()) - referenced_varnames
            for varname in names_to_delete:
                del soln_variables[varname]

        results.solution.insert(soln)

        self.results = results
        # Done with the model object; free up some memory.
        self._last_native_var_idx = -1
        self._range_con_var_pairs = []

        # finally, clean any temporary files registered with the temp file
        # manager, created populated *directly* by this plugin.
        pyutilib.services.TempfileManager.pop(remove=not self._keepfiles)

        # let the base class deal with returning results.
        return OptSolver._postsolve(self)


class GurobiDirect(DirectSolver):
    alias('gurobi_direct', doc='Direct python interface to gurobi')

    def __init__(self, **kwds):
        try:
            import gurobipy
            self._gurobipy = gurobipy
            self._gurobi_python_api_exists = True
            self._gurobi_version = self._gurobipy.gurobi.version()
            while len(self._gurobi_version) < 4:
                self._gurobi_version += (0,)
            self._gurobi_version = self._gurobi_version[:4]
            self._gurobi_version_major = self._gurobi_version[0]
        except ImportError:
            self._gurobi_python_api_exists = False
        except Exception as e:
            # other forms of exceptions can be thrown by the gurobi python
            # import. for example, a gurobipy.GurobiError exception is thrown
            # if all tokens for Gurobi are already in use. assuming, of
            # course, the license is a token license. unfortunately, you can't
            # import without a license, which means we can't test for the
            # exception above!
            print("Import of gurobipy failed - gurobi message=" + str(e) + "\n")
            self._gurobi_python_api_exists = False

        kwds['type'] = 'gurobi_direct'
        super(GurobiDirect, self).__init__(**kwds)

        self._range_constraints = set()

        # Note: Undefined capabilites default to None
        self._capabilities = Options()
        self._capabilities.linear = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = True
        self._capabilities.integer = True
        self._capabilities.sos1 = True
        self._capabilities.sos2 = True

    def _get_version(self):
        if self._gurobi_version is None:
            return _extract_version('')
        return self._gurobi_version

    def _presolve(self, *args, **kwds):
        self._solver_model = self._gurobipy.Model()
        warmstart_flag = kwds.pop('warmstart', False)

        super(GurobiDirect, self)._presolve(*args, **kwds)

        if warmstart_flag:
            self._warm_start()

        if self._log_file is None:
            self._log_file = pyutilib.services.TempfileManager.create_tempfile(suffix='.gurobi.log')

    def _apply_solver(self):
        if self._tee:
            self._solver_model.setParam('OutputFlag', 1)
        else:
            self._solver_model.setParam('OutputFlag', 0)

        self._solver_model.setParam('LogFile', self._log_file)

        if self._keepfiles:
            print("Solver log file: "+self._log_file)

        #Options accepted by gurobi (case insensitive):
        #['Cutoff', 'IterationLimit', 'NodeLimit', 'SolutionLimit', 'TimeLimit',
        # 'FeasibilityTol', 'IntFeasTol', 'MarkowitzTol', 'MIPGap', 'MIPGapAbs',
        # 'OptimalityTol', 'PSDTol', 'Method', 'PerturbValue', 'ObjScale', 'ScaleFlag',
        # 'SimplexPricing', 'Quad', 'NormAdjust', 'BarIterLimit', 'BarConvTol',
        # 'BarCorrectors', 'BarOrder', 'Crossover', 'CrossoverBasis', 'BranchDir',
        # 'Heuristics', 'MinRelNodes', 'MIPFocus', 'NodefileStart', 'NodefileDir',
        # 'NodeMethod', 'PumpPasses', 'RINS', 'SolutionNumber', 'SubMIPNodes', 'Symmetry',
        # 'VarBranch', 'Cuts', 'CutPasses', 'CliqueCuts', 'CoverCuts', 'CutAggPasses',
        # 'FlowCoverCuts', 'FlowPathCuts', 'GomoryPasses', 'GUBCoverCuts', 'ImpliedCuts',
        # 'MIPSepCuts', 'MIRCuts', 'NetworkCuts', 'SubMIPCuts', 'ZeroHalfCuts', 'ModKCuts',
        # 'Aggregate', 'AggFill', 'PreDual', 'DisplayInterval', 'IISMethod', 'InfUnbdInfo',
        # 'LogFile', 'PreCrush', 'PreDepRow', 'PreMIQPMethod', 'PrePasses', 'Presolve',
        # 'ResultFile', 'ImproveStartTime', 'ImproveStartGap', 'Threads', 'Dummy', 'OutputFlag']
        for key, option in self.options.items():
            self._solver_model.setParam(key, option)

        if self._gurobi_version_major >= 5:
            for suffix in self._suffixes:
                if re.match(suffix, "dual"):
                    self._solver_model.setParam(gurobi_direct._gurobi_module.GRB.Param.QCPDual, 1)

        self._solver_model.optimize()

        self._solver_model.setParam('LogFile', 'default')

        return Bunch(rc=None, log=None)

    def _get_expr_from_pyomo_expr(self, expr):
        repn = generate_canonical_repn(expr)

        degree = canonical_degree(repn)
        if degree not in [0,1,2]:
            raise ValueError('GurobiDirect only supports linear and quadratic constraints\n{0}'.format(expr))

        if isinstance(repn, LinearCanonicalRepn):
            # list(map(self._referenced_variable_ids.add, map(id, repn.variables)))
            return repn.constant + sum(coeff*self._pyomo_var_to_solver_var_map[id(repn.variables[i])] for i, coeff in
                                       enumerate(repn.linear))

        else:
            # list(map(self._referenced_variable_ids.add, map(id, repn[-1].values())))
            new_expr = 0
            if 0 in repn:
                new_expr += repn[0][None]

            if 1 in repn:
                for ndx, coeff in repn[1].items():
                    new_expr += coeff * self._pyomo_var_to_solver_var_map[id(repn[-1][ndx])]

            if 2 in repn:
                if self._gurobi_version_major < 5:
                    raise ValueError('The gurobi direct plugin does not handle quadratic constraint ' +
                                     'expressions for \nGurobi versions < 5. Current ' +
                                     'version: {0}'.format(self._gurobipy.gurobi.version()))
                for key, coeff in repn[2].items():
                    tmp_expr = coeff
                    for ndx, power in key.items():
                        for i in range(power):
                            tmp_expr *= self._pyomo_var_to_solver_var_map[id(repn[-1][ndx])]
                    new_expr += tmp_expr

            return new_expr

    def _add_var(self, var):
        varname = self._symbol_map.getSymbol(var, self._labeler)
        domain = var.domain
        vtype = self._gurobi_vtype_from_var(var)
        lb = value(var.lb)
        ub = value(var.ub)
        if lb is None:
            lb = -self._gurobipy.GRB.INFINITY
        if ub is None:
            ub = self._gurobipy.GRB.INFINITY

        gurobipy_var = self._solver_model.addVar(lb=lb, ub=ub, vtype=vtype, name=varname)

        self._pyomo_var_to_solver_var_map[id(var)] = gurobipy_var

        if var.is_fixed():
            gurobipy_var.setAttr('lb', var.value)
            gurobipy_var.setAttr('ub', var.value)

    def _compile_instance(self, model):
        if isinstance(model, IBlockStorage):
            # BIG HACK (see pyomo.core.kernel write function)
            if not hasattr(model, "._symbol_maps"):
                setattr(model, "._symbol_maps", {})
            getattr(model,
                    "._symbol_maps")[self._smap_id] = self._symbol_map
        else:
            model.solutions.add_symbol_map(self._symbol_map)
        self._add_block(model)

    def _add_block(self, block):
        for var in block.component_data_objects(ctype=pyomo.core.base.var.Var, descend_into=True, active=True):
            self._add_var(var)
        self._solver_model.update()

        for con in block.component_data_objects(ctype=pyomo.core.base.constraint.Constraint,
                                                descend_into=True, active=True):
            self._add_constraint(con)

        for con in block.component_data_objects(ctype=pyomo.core.base.sos.SOSConstraint,
                                                descend_into=True, active=True):
            self._add_sos_constraint(con)

        self._compile_objective()

    def _add_constraint(self, con):
        if not con.active:
            return None

        if is_fixed(con.body):
            if self._skip_trivial_constraints:
                return None
            raise ValueError('Expression for constraint body is a constant: {0} \n'.format(con) +
                             'To suppress this error, add skip_trivial_constraints=True to the call to solve.')

        conname = self._symbol_map.getSymbol(con, self._labeler)
        gurobi_expr = self._get_expr_from_pyomo_expr(con.body)

        if con.has_lb():
            if not is_fixed(con.lower):
                raise ValueError('Lower bound of constraint {0} is not constant.'.format(con))
        if con.has_ub():
            if not is_fixed(con.upper):
                raise ValueError('Upper bound of constraint {0} is not constant.'.format(con))

        if con.equality:
            gurobipy_con = self._solver_model.addConstr(gurobi_expr==value(con.lower), name=conname)
        elif con.has_lb() and con.has_ub():
            if 'slack' in self._suffixes:
                raise ValueError('GurobiDirect does not support range constraints and slack suffixes. \nIf you want ' +
                                 'slack information, please split this into two constraints: \n{0}'.format(con))
            gurobipy_con = self._solver_model.addRange(gurobi_expr, value(con.lower), value(con.upper), name=conname)
            self._range_constraints.add(con)
        elif con.has_lb():
            gurobipy_con = self._solver_model.addConstr(gurobi_expr>=value(con.lower), name=conname)
        elif con.has_ub():
            gurobipy_con = self._solver_model.addConstr(gurobi_expr<=value(con.upper), name=conname)
        else:
            if self._skip_trivial_constraints:
                return None
            raise ValueError('Constraint does not have a lower or an upper bound: {0} \n'.format(con) +
                             'To suppress this error, add skip_trival_constraints=True to the call to solve.')

        self._pyomo_con_to_solver_con_map[id(con)] = gurobipy_con

    def _add_sos_constraint(self, con):
        if not con.active:
            return None

        conname = self._symbol_map.getSymbol(con, self._labeler)
        level = con.level
        if level == 1:
            sos_type = self._gurobipy.GRB.SOS_TYPE1
        elif level == 2:
            sos_type = self._gurobipy.GRB.SOS_TYPE2
        else:
            raise ValueError('Solver does not support SOS level {0} constraints'.format(level))

        gurobi_vars = []
        weights = []

        for v, w in con.get_items():
            gurobi_vars.append(self._pyomo_var_to_solver_var_map[id(v)])
            weights.append(w)

        gurobipy_con = self._solver_model.addSOS(sos_type, gurobi_vars, weights)
        self._pyomo_con_to_solver_con_map[id(con)] = gurobipy_con

    def _gurobi_vtype_from_var(self, var):
        """
        This function takes a pyomo variable and returns the appropriate gurobi variable type
        :param var: pyomo.core.base.var.Var
        :return: gurobipy.GRB.CONTINUOUS or gurobipy.GRB.BINARY or gurobipy.GRB.INTEGER
        """
        if var.is_binary():
            vtype = self._gurobipy.GRB.BINARY
        elif var.is_integer():
            vtype = self._gurobipy.GRB.INTEGER
        elif var.is_continuous():
            vtype = self._gurobipy.GRB.CONTINUOUS
        else:
            raise ValueError('Variable domain type is not recognized for {0}'.format(var.domain))
        return vtype

    def _compile_objective(self):
        obj_counter = 0

        for obj in self._pyomo_model.component_data_objects(ctype=pyomo.core.base.objective.Objective, descend_into=True, active=True):
            obj_counter += 1
            if obj_counter > 1:
                raise ValueError('Multiple active objectives found. Solver only handles one active objective')

            if obj.sense == pyomo.core.kernel.minimize:
                sense = self._gurobipy.GRB.MINIMIZE
            elif obj.sense == pyomo.core.kernel.maximize:
                sense = self._gurobipy.GRB.MAXIMIZE
            else:
                raise ValueError('Objective sense is not recognized: {0}'.format(obj.sense))

            self._solver_model.setObjective(self._get_expr_from_pyomo_expr(obj.expr), sense=sense)
            self._objective_label = self._symbol_map.getSymbol(obj, self._labeler)

    def _postsolve(self):
        # the only suffixes that we extract from GUROBI are
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
                if len(self._range_constraints) != 0:
                    err_msg = ('GurobiDirect does not support range constraints and slack suffixes. \nIf you want ' +
                               'slack information, please split up the following constraints:\n')
                    for con in self._range_constraints:
                        err_msg += '{0}\n'.format(con)
                    raise ValueError(err_msg)
            if re.match(suffix,"rc"):
                extract_reduced_costs = True
                flag=True
            if not flag:
                raise RuntimeError(
                    "***The gurobi_direct solver plugin "
                    "cannot extract solution suffix="+suffix)

        gprob = self._solver_model

        if gprob.getAttr(self._gurobipy.GRB.Attr.IsMIP):
            extract_reduced_costs = False
            extract_duals = False

        pvars = gprob.getVars()
        cons = gprob.getConstrs()
        qcons = []
        if self._gurobi_version_major >= 5:
            qcons = gprob.getQConstrs()

        self.results = SolverResults()
        soln = Solution()

        # cache the variable and constraint dictionaries -
        # otherwise, each invocation will include a lookup in a
        # MapContainer, which is extremely expensive.
        soln_variables = soln.variable
        soln_constraints = soln.constraint

        self.results.solver.name = ("Gurobi %s.%s%s"
                       % self._gurobipy.gurobi.version())
        self.results.solver.wallclock_time = gprob.Runtime

        if gprob.Status == 1: # problem is loaded, but no solution
            self.results.solver.termination_message = "Model is loaded, but no solution information is available."
            self.results.solver.termination_condition = TerminationCondition.other
        if gprob.Status == 2: # optimal
            self.results.solver.termination_message = "Model was solved to optimality (subject to tolerances)."
            self.results.solver.termination_condition = TerminationCondition.optimal
        elif gprob.Status == 3: # infeasible
            self.results.solver.termination_message = "Model was proven to be infeasible."
            self.results.solver.termination_condition = TerminationCondition.infeasible
        elif gprob.Status == 4: # infeasible or unbounded
            self.results.solver.termination_message = "Model was proven to be either infeasible or unbounded."
            self.results.solver.termination_condition = TerminationCondition.infeasible # picking one of the pre-specified Pyomo termination conditions - we don't have either-or.
        elif gprob.Status == 5: # unbounded
            self.results.solver.termination_message = "Model was proven to be unbounded."
            self.results.solver.termination_condition = TerminationCondition.unbounded
        elif gprob.Status == 6: # cutoff
            self.results.solver.termination_message = "Optimal objective for model was proven to be worse than the value specified in the Cutoff parameter."
            self.results.solver.termination_condition = TerminationCondition.minFunctionValue
        elif gprob.Status == 7: # iteration limit
            self.results.solver.termination_message = "Optimization terminated because the total number of simplex or barrier iterations exceeded specified limits."
            self.results.solver.termination_condition = TerminationCondition.maxIterations
        elif gprob.Status == 8: # node limit
            self.results.solver.termination_message = "Optimization terminated because the total number of branch-and-cut nodes exceeded specified limits."
            self.results.solver.termination_condition = TerminationCondition.maxEvaluations
        elif gprob.Status == 9: # time limit
            self.results.solver.termination_message = "Optimization terminated because the total time expended exceeded specified limits."
            self.results.solver.termination_condition = TerminationCondition.maxTimeLimit
        elif gprob.Status == 10: # solution limit
            self.results.solver.termination_message = "Optimization terminated because the number of solutions found reached specified limits."
            self.results.solver.termination_condition = TerminationCondition.other
        elif gprob.Status == 11: # interrupted
            self.results.solver.termination_message = "Optimization was terminated by the user."
            self.results.solver.termination_condition = TerminationCondition.other
        elif gprob.Status == 12: # numeric issues
            self.results.solver.termination_message = "Optimization was terminated due to unrecoverable numerical difficulties."
            self.results.solver.termination_condition = TerminationCondition.other
        elif gprob.Status == 13: # suboptimal
            self.results.solver.termination_message = "Optimization was unable to satisfy optimality tolerances; returned solution is sub-optimal."
            self.results.solver.termination_condition = TerminationCondition.other
        else: # unknown
            self.results.solver.termination_message = "Unknown termination condition received following optimization."
            self.results.solver.termination_condition = TerminationCondition.other

        self.results.problem.name = gprob.ModelName
        if gprob.ModelSense == 1:  # minimizing
            self.results.problem.sense = pyomo.core.kernel.minimize
            try:
                self.results.problem.upper_bound = gprob.ObjVal
            except:
                pass
            try:
                self.results.problem.lower_bound = gprob.ObjBound
            except:
                pass
        elif gprob.ModelSense == -1:  # maximizing
            self.results.problem.sense = pyomo.core.kernel.maximize
            try:
                self.results.problem.upper_bound = gprob.ObjBound
            except:
                pass
            try:
                self.results.problem.lower_bound = gprob.ObjVal
            except:
                pass
        else:
            raise RuntimeError('Unrecognized gurobi objective sense: {0}'.format(gprob.ModelSense))

        try:
            soln.gap = self.results.problem.upper_bound - self.results.problem.lower_bound
        except:
            soln.gap = None

        self.results.problem.number_of_constraints = gprob.NumConstrs + gprob.NumQConstrs + gprob.NumSOS
        self.results.problem.number_of_nonzeros = gprob.NumNZs
        self.results.problem.number_of_variables = gprob.NumVars
        self.results.problem.number_of_binary_variables = gprob.NumBinVars
        self.results.problem.number_of_integer_variables = gprob.NumIntVars
        self.results.problem.number_of_continuous_variables = gprob.NumVars - gprob.NumIntVars - gprob.NumBinVars
        self.results.problem.number_of_objectives = 1
        self.results.problem.number_of_solutions = gprob.SolCount

        status = self._solver_model.Status
        if self._gurobipy.GRB.OPTIMAL == status:
            soln.status = SolutionStatus.optimal
        elif self._gurobipy.GRB.INFEASIBLE == status:
            soln.status = SolutionStatus.infeasible
        elif self._gurobipy.GRB.CUTOFF == status:
            soln.status = SolutionStatus.other
        elif self._gurobipy.GRB.INF_OR_UNBD == status:
            soln.status = SolutionStatus.other
        elif self._gurobipy.GRB.INTERRUPTED == status:
            soln.status = SolutionStatus.other
        elif self._gurobipy.GRB.LOADED == status:
            soln.status = SolutionStatus.other
        elif self._gurobipy.GRB.SUBOPTIMAL == status:
            soln.status = SolutionStatus.other
        elif self._gurobipy.GRB.UNBOUNDED == status:
            soln.status = SolutionStatus.other
        elif self._gurobipy.GRB.ITERATION_LIMIT == status:
            soln.status = SolutionStatus.stoppedByLimit
        elif self._gurobipy.GRB.NODE_LIMIT == status:
            soln.status = SolutionStatus.stoppedByLimit
        elif self._gurobipy.GRB.SOLUTION_LIMIT == status:
            soln.status = SolutionStatus.stoppedByLimit
        elif self._gurobipy.GRB.TIME_LIMIT == status:
            soln.status = SolutionStatus.stoppedByLimit
        elif self._gurobipy.GRB.NUMERIC == status:
            soln.status = SolutionStatus.error
        else:
            raise RuntimeError("Unknown solution status returned by "
                               "Gurobi solver")

        # if a solve was stopped by a limit, we still need to check to
        # see if there is a solution available - this may not always
        # be the case, both in LP and MIP contexts.
        if (soln.status == SolutionStatus.optimal) or \
           ((soln.status == SolutionStatus.stoppedByLimit) and \
            (gprob.SolCount > 0)):

            soln.objective[self._objective_label] = {'Value': gprob.ObjVal}

            for var in self._pyomo_var_to_solver_var_map.values():
                soln_variables[var.VarName] = {"Value": var.x}

            if extract_reduced_costs:
                for var in self._pyomo_var_to_solver_var_map.values():
                    soln_variables[var.VarName]["Rc"] = var.Rc

            if extract_duals or extract_slacks:
                for con in cons:
                    soln_constraints[con.ConstrName] = {}
                for con in qcons:
                    soln_constraints[con.QCName] = {}

            if extract_duals:
                for con in cons:
                    # Pi attributes in Gurobi are the constraint duals
                    soln_constraints[con.ConstrName]["Dual"] = con.Pi
                for con in qcons:
                    # QCPI attributes in Gurobi are the constraint duals
                    soln_constraints[con.QCName]["Dual"] = con.QCPi

            if extract_slacks:
                for con in cons:
                    soln_constraints[con.ConstrName]["Slack"] = con.Slack
                for con in qcons:
                    soln_constraints[con.QCName]["Slack"] = con.QCSlack

        self.results.solution.insert(soln)

        # finally, clean any temporary files registered with the temp file
        # manager, created populated *directly* by this plugin.
        pyutilib.services.TempfileManager.pop(remove=not self._keepfiles)

        return super(GurobiDirect, self)._postsolve()

    def available(self, exception_flag=True):
        """True if the solver is available."""

        if exception_flag is False:
            return self._gurobi_python_api_exists
        else:
            if self._gurobi_python_api_exists is False:
                raise pyutilib.common.ApplicationError("No Gurobi <-> PYthon bindings available - Gurobi direct solver "
                                                       "functionality is not available.")
            else:
                return True

    def warm_start_capable(self):
        return True

    def _warm_start(self):
        for var_id, gurobipy_var in self._pyomo_var_to_solver_var_map.items():
            varname = self._symbol_map.byObject[var_id]
            pyomo_var = self._symbol_map.bySymbol[varname]()
            if pyomo_var.value is not None:
                gurobipy_var.setAttr(self._gurobipy.GRB.Attr.Start, pyomo_var.value)
