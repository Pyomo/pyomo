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
import re
import itertools
import math
import sys

from six import itervalues, iteritems, advance_iterator

logger = logging.getLogger('pyomo.solvers')

_gurobi_version = None
try:
    # import all the glp_* functions
    from gurobipy import *
    # create a version tuple of length 4
    _gurobi_version = gurobi.version()
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
    # course, the license is a token license. unfortunately, you 
    # can't import without a license, which means we can't test 
    # for the exception above!
    print("Import of gurobipy failed - gurobi message="+str(e)+"\n")
    gurobi_python_api_exists = False

from pyutilib.misc import Bunch, Options
from pyomo.util.plugin import alias
from pyutilib.services import TempfileManager

from pyomo.opt.base import *
from pyomo.opt.base.solvers import _extract_version
from pyomo.opt.results import *
from pyomo.opt.solver import *
from pyomo.core.base import SymbolMap, BasicSymbolMap, NumericLabeler, ComponentMap, TextLabeler
from pyomo.core.base.numvalue import value
from pyomo.core.base.block import active_components, active_components_data

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

        sos_items = soscondata.get_items()
        level = soscondata.get_level()

        if len(sos_items) == 0:
            return

        self.block_cntr += 1
        varnames = self.varnames[self.block_cntr] = []
        varids = self.varids[self.block_cntr] = []
        weights = self.weights[self.block_cntr] = []
        if level == 1:
            self.sosType[self.block_cntr] = GRB.SOS_TYPE1
        elif level == 2:
            self.sosType[self.block_cntr] = GRB.SOS_TYPE2
        else:
            raise ValueError("Unsupported SOSConstraint level %s" % level)

        self.sosName[self.block_cntr] = symbol_map.getSymbol(soscondata,labeler)
        
        for vardata, weight in sos_items:
            if vardata.fixed:
                raise RuntimeError("SOSConstraint '%s' includes a fixed variable '%s'. "
                                   "This is currently not supported. Deactivate this constraint " 
                                   "in order to proceed." % (soscondata.cname(True), vardata.cname(True)))
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

    alias('_gurobi_direct', doc='Direct Python interface to the Gurobi optimization solver.')

    def __init__(self, **kwds):
        #
        # Call base class constructor
        #
        kwds['type'] = 'gurobi_direct'
        OptSolver.__init__(self, **kwds)

        self._model = None
        
        # a dictionary that maps pyomo _VarData labels to the corresponding Gurobi 
        # variable object. created each time _populate_gurobi_instance is called.
        self._pyomo_gurobi_variable_map = None

        # NOTE: eventually both of the following attributes should be migrated
        # to a common base class.  Is the current solve warm-started?  A
        # transient data member to communicate state information across the
        # _presolve, _apply_solver, and _postsolve methods.
        self.warm_start_solve = False

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

    def version(self):
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

    def _populate_gurobi_instance ( self, pyomo_instance ):

        from pyomo.core.base import Var, Objective, Constraint, ConstraintList, IntegerSet, BooleanSet, SOSConstraint
        from pyomo.repn import canonical_is_constant
        from pyomo.core import LinearCanonicalRepn

        try:
            grbmodel = Model(name=pyomo_instance.name)
        except Exception:
            e = sys.exc_info()[1]
            msg = 'Unable to create Gurobi model.  Have you installed the Python'\
            '\n       bindings for Gurobi?\n\n\tError message: %s'
            raise Exception(msg % e)

        if self.symbolic_solver_labels is True:
            labeler = TextLabeler()
        else:
            labeler = NumericLabeler('x')
        # cache to avoid dictionary getitem calls in the loops below.
        self_symbol_map = self._symbol_map = SymbolMap(pyomo_instance)
        # we use this when iterating over the constraints because it will have a much smaller hash
        # table, we also use this for the warm start code after it is cleaned to only contain
        # variables referenced in the constraints
        self_variable_symbol_map = self._variable_symbol_map = BasicSymbolMap()
        var_symbol_pairs = []

        pyomo_gurobi_variable_map = {} # maps _VarData labels to the corresponding Gurobi variable object

        self._referenced_variable_ids.clear()

        # cache to avoid dictionary getitem calls in the loop below.
        grb_infinity = GRB.INFINITY

        for block in pyomo_instance.all_blocks():
            for var_value in active_components_data(block, Var):
    
                lb = -grb_infinity
                ub = grb_infinity
    
                if var_value.lb is not None:
                    lb = value(var_value.lb)
                if var_value.ub is not None:
                    ub = value(var_value.ub)
    
                # _VarValue objects will not be in the symbol map yet, so avoid some checks.
                var_value_label = self_symbol_map.createSymbol(var_value, labeler)
                var_symbol_pairs.append((var_value, var_value_label))
                
                # be sure to impart the integer and binary nature of any variables
                if var_value.is_integer():
                    var_type = GRB.INTEGER
                elif var_value.is_binary():
                    var_type = GRB.BINARY
                elif var_value.is_continuous():
                    var_type = GRB.CONTINUOUS
                else:
                    raise TypeError("Invalid domain type for variable with name '%s'. "
                                    "Variable is not continuous, integer, or binary.")
 
                pyomo_gurobi_variable_map[var_value_label] = grbmodel.addVar(lb=lb, \
                                                                             ub=ub, \
                                                                             vtype=var_type, \
                                                                             name=var_value_label)

        self_variable_symbol_map.updateSymbols(var_symbol_pairs)

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
                                          pyomo_gurobi_variable_map,
                                          soscondata)

            # Objective
            for obj_data in active_components_data(block, Objective):

                if objective_cntr > 1:
                    raise ValueError("Multiple active objectives found on Pyomo instance '%s'. "
                                     "Solver '%s' will only handle a single active objective" \
                                     % (pyomo_instance.cname(True), self.type))

                sense = GRB_MIN if (obj_data.is_minimizing()) else GRB_MAX
                grbmodel.ModelSense = sense
                obj_expr = LinExpr()

                obj_repn = block_canonical_repn.get(obj_data)
                if obj_repn is None:
                    raise ValueError("No entry found in canonical_repn ComponentMap on "
                                     "block %s for active objective with name %s. "
                                     "Did you forget to preprocess?"
                                     % (block.cname(True), obj_data.cname(True)))

                if isinstance(obj_repn, LinearCanonicalRepn):

                    if obj_repn.constant != None:
                        obj_expr.addConstant(obj_repn.constant)

                    if obj_repn.linear != None:

                        for i in xrange(len(obj_repn.linear)):
                            var_coefficient = obj_repn.linear[i]
                            var_value = obj_repn.variables[i]
                            self._referenced_variable_ids.add(id(var_value))
                            label = self_variable_symbol_map.getSymbol(var_value)
                            obj_expr.addTerms(var_coefficient, pyomo_gurobi_variable_map[label])
                else:

                    if 0 in obj_repn: # constant term
                        obj_expr.addConstant(obj_repn[0][None])

                    if 1 in obj_repn: # first-order terms
                        hash_to_variable_map = obj_repn[-1]
                        for var_hash, var_coefficient in iteritems(obj_repn[1]):
                            vardata = hash_to_variable_map[var_hash]
                            self._referenced_variable_ids.add(id(vardata))
                            label = self_variable_symbol_map.getSymbol(vardata)
                            obj_expr.addTerms(var_coefficient, pyomo_gurobi_variable_map[label])

                    if 2 in obj_repn:
                        obj_expr = QuadExpr(obj_expr)
                        hash_to_variable_map = obj_repn[-1]
                        for quad_repn, coef in iteritems(obj_repn[2]):
                            gurobi_expr = QuadExpr(coef)
                            for var_hash, exponent in iteritems(quad_repn):
                                vardata = hash_to_variable_map[var_hash]
                                self._referenced_variable_ids.add(id(vardata))
                                gurobi_var = pyomo_gurobi_variable_map[self_variable_symbol_map.getSymbol(vardata)]
                                gurobi_expr *= gurobi_var
                                if exponent == 2:
                                    gurobi_expr *= gurobi_var
                            obj_expr += gurobi_expr

                # need to cache the objective label, because the GUROBI python interface doesn't track this.
                # _ObjectiveData objects will not be in the symbol map yet, so avoid some checks.
                self._objective_label = self_symbol_map.createSymbol(obj_data, labeler)

                grbmodel.setObjective(obj_expr, sense=sense)

            # Constraint
            for constraint in active_components(block,Constraint):
                if constraint.trivial: 
                    continue
    
                for constraint_data in itervalues(constraint):
    
                    if not constraint_data.active: 
                        continue
                    elif constraint_data.lower is None and constraint_data.upper is None:
                        continue  # not binding at all, don't bother

                    con_repn = block_canonical_repn.get(constraint_data)
                    if constraint_repn is None:
                        raise ValueError("No entry found in canonical_repn ComponentMap on "
                                         "block %s for active constraint with name %s. "
                                         "Did you forget to preprocess?"
                                         % (block.cname(True), constraint_data.cname(True)))

                    offset = 0.0
                    # _ConstraintData objects will not be in the symbol map yet, so avoid some checks.
                    constraint_label = self_symbol_map.createSymbol(constraint_data, labeler)

                    if isinstance(con_repn, LinearCanonicalRepn):

                        if con_repn.constant != None:
                            offset = con_repn.constant
                        expr = LinExpr() + offset

                        if con_repn.linear != None:
    
                            linear_coefs = list()
                            linear_vars = list()

                            for i in xrange(len(con_repn.linear)):

                                var_coefficient = con_repn.linear[i]
                                var_value = con_repn.variables[i]
                                self._referenced_variable_ids.add(id(var_value))
                                label = self_variable_symbol_map.getSymbol(var_value)
                                linear_coefs.append( var_coefficient )
                                linear_vars.append( pyomo_gurobi_variable_map[label] )
    
                            expr += LinExpr(linear_coefs, linear_vars)

                    else:                        

                        if 0 in con_repn:
                            offset = con_repn[0][None]
                        expr = LinExpr() + offset

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
    
                            expr += LinExpr(linear_coefs, linear_vars)

                        if 2 in con_repn: # quadratic constraint
                            if _GUROBI_VERSION_MAJOR < 5:
                                raise ValueError("The gurobi_direct plugin does not handle quadratic constraint expressions\
                            \nfor Gurobi major versions < 5. Current version: Gurobi %s.%s%s" % gurobi.version())
                            expr = QuadExpr(expr)
                            hash_to_variable_map = con_repn[-1]
                            for quad_repn, coef in iteritems(con_repn[2]):
                                gurobi_expr = QuadExpr(coef)
                                for var_hash, exponent in iteritems(quad_repn):
                                    vardata = hash_to_variable_map[var_hash]
                                    self._referenced_variable_ids.add(id(vardata))
                                    gurobi_var = pyomo_gurobi_variable_map[self_variable_symbol_map.getSymbol(vardata)]
                                    gurobi_expr *= gurobi_var 
                                    if exponent == 2:
                                        gurobi_expr *= gurobi_var
                                expr += gurobi_expr

                    if constraint_data._equality:
                        sense = GRB.EQUAL    # Fixed
                        bound = constraint_data.lower()
                        grbmodel.addConstr(lhs=expr, 
                                           sense=sense, 
                                           rhs=bound, 
                                           name=constraint_label )
                    else:
                        # L <= body <= U
                        if (constraint_data.upper is not None) and (constraint_data.lower is not None):
                            grb_con = grbmodel.addRange(expr, constraint_data.lower(), constraint_data.upper(), constraint_label)
                            _self_range_con_var_pairs.append((grb_con,range_var_idx))
                            range_var_idx += 1
                        # body <= U
                        elif constraint_data.upper is not None:
                            bound = constraint_data.upper()
                            if bound < float('inf'):
                                grbmodel.addConstr(
                                    lhs=expr,
                                    sense=GRB.LESS_EQUAL,
                                    rhs=bound,
                                    name=constraint_label
                                    )
                        # L <= body            
                        else:
                            bound = constraint_data.lower()
                            if bound > -float('inf'):
                                grbmodel.addConstr(
                                    lhs=expr,
                                    sense=GRB.GREATER_EQUAL,
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
            vardata = self._variable_symbol_map.bySymbol[varname]
            if vardata.fixed:
                if not self.output_fixed_variable_bounds:
                    raise ValueError("Encountered a fixed variable (%s) inside an active objective "
                                     "or constraint expression on model %s, which is usually indicative of "
                                     "a preprocessing error. Use the IO-option 'output_fixed_variable_bounds=True' "
                                     "to suppress this error and fix the variable by overwriting its bounds in "
                                     "the Gurobi instance."
                                     % (vardata.cname(True),pyomo_instance.cname(True),))

                grbvar = pyomo_gurobi_variable_map[varname]
                grbvar.setAttr(GRB.attr.ub, vardata.value)
                grbvar.setAttr(GRB.attr.lb, vardata.value)

        grbmodel.update()

        self._gurobi_instance = grbmodel
        self._pyomo_gurobi_variable_map = pyomo_gurobi_variable_map

    def warm_start_capable(self):

        return True

    def warm_start(self, instance):

        for symbol, vardata in iteritems(self._variable_symbol_map.bySymbol):
            if vardata.value is not None:
                self._pyomo_gurobi_variable_map[symbol].setAttr(GRB.attr.Start, vardata.value)
                
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

        model = args[0]
        if len(args) != 1:
            msg = "The gurobi_direct plugin method '_presolve' must be supplied "\
                  "a single problem instance - %s were supplied"
            raise ValueError(msg % len(args))
        elif not isinstance(model, Model):
            raise ValueError("The problem instance supplied to the "            \
                 "gurobi_direct plugin '_presolve' method must be of type 'Model'")

        self._populate_gurobi_instance( model )
        grbmodel = self._gurobi_instance

        # Clean up the symbol map to only contain variables referenced in the constraints
        # **NOTE**: The warmstart method (if called below), relies on a "clean" symbol map
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
            grbmodel.write( fname )

        if self.warm_start_solve is True:

            if len(args) != 1:
                msg = "The gurobi_direct _presolve method can only handle a single"\
                      "problem instance - %s were supplied"
                raise ValueError(msg % len(args))

            self.warm_start( model )

        # This does not use weak references so we need to release references to external model variables
        del self._variable_symbol_map


    def _apply_solver(self):
        # TODO apply appropriate user-specified parameters

        prob = self._gurobi_instance

        if self.tee:
            prob.setParam( 'OutputFlag', self.tee )
        else:
            prob.setParam( 'OutputFlag', 0)
        
        if self.keepfiles == True:
            log_file = TempfileManager.create_tempfile(suffix = '.gurobi.log')
            print("Solver log file: " + log_file)
            prob.setParam('LogFile', log_file)

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
                if v.vType != GRB.CONTINUOUS:
                    v.vType = GRB.CONTINUOUS
            prob.update()

        if _GUROBI_VERSION_MAJOR >= 5:
            for suffix in self.suffixes:
                if re.match(suffix,"dual"):
                    prob.setParam(GRB.Param.QCPDual,1)

        # Actually solve the problem.
        prob.optimize()

        # FIXME: can we get a return code indicating if Gurobi had a
        # significant failure?
        return Bunch(rc=None, log=None)


    def _gurobi_get_solution_status ( self ):
        status = self._gurobi_instance.Status
        if   GRB.OPTIMAL         == status: return SolutionStatus.optimal
        elif GRB.INFEASIBLE      == status: return SolutionStatus.infeasible
        elif GRB.CUTOFF          == status: return SolutionStatus.other
        elif GRB.INF_OR_UNBD     == status: return SolutionStatus.other
        elif GRB.INTERRUPTED     == status: return SolutionStatus.other
        elif GRB.LOADED          == status: return SolutionStatus.other
        elif GRB.SUBOPTIMAL      == status: return SolutionStatus.other
        elif GRB.UNBOUNDED       == status: return SolutionStatus.other
        elif GRB.ITERATION_LIMIT == status: return SolutionStatus.stoppedByLimit
        elif GRB.NODE_LIMIT      == status: return SolutionStatus.stoppedByLimit
        elif GRB.SOLUTION_LIMIT  == status: return SolutionStatus.stoppedByLimit
        elif GRB.TIME_LIMIT      == status: return SolutionStatus.stoppedByLimit
        elif GRB.NUMERIC         == status: return SolutionStatus.error
        raise RuntimeError('Unknown solution status returned by Gurobi solver')

    def _postsolve(self):

        # the only suffixes that we extract from GUROBI are
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
                raise RuntimeError("***The gurobi_direct solver plugin cannot extract solution suffix="+suffix)

        gprob = self._gurobi_instance

        if (gprob.getAttr(GRB.Attr.IsMIP)):
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

        # cache the variable and constraint dictionaries - otherwise,
        # each invocation will include a lookup in a MapContainer, 
        # which is extremely expensive.
        soln_variables = soln.variable
        soln_constraints = soln.constraint

        solver.name = "Gurobi %s.%s%s" % gurobi.version()
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
            solver.termination_condition = TerminationCondition.infeasible # picking one of the pre-specified Pyomo termination conditions - we don't have either-or.
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
        if problem.sense == GRB_MAX: problem.sense = ProblemSense.maximize

        soln.status = self._gurobi_get_solution_status()
        soln.gap = None # until proven otherwise

        # if a solve was stopped by a limit, we still need to check to see if
        # there is a solution available - this may not always be the case, both
        # in LP and MIP contexts.
        if (soln.status == SolutionStatus.optimal) or ((soln.status == SolutionStatus.stoppedByLimit) and gprob.SolCount > 0):

            obj_val = gprob.ObjVal

            if problem.number_of_binary_variables + problem.number_of_integer_variables > 0:
                obj_bound = gprob.ObjBound

            if problem.sense == ProblemSense.minimize:
                if problem.number_of_binary_variables + problem.number_of_integer_variables == 0:
                    problem.upper_bound = obj_val
                else:
                    problem.lower_bound = obj_bound
                    problem.upper_bound = obj_val
            else:

                if problem.number_of_binary_variables + problem.number_of_integer_variables == 0:
                    problem.lower_bound = obj_val
                else:
                    problem.upper_bound = obj_bound
                    problem.lower_bound = obj_val

            soln.objective[self._objective_label].value = obj_val
            if problem.number_of_binary_variables + problem.number_of_integer_variables == 0:
                soln.gap = None
            else:
                soln.gap = math.fabs(obj_val - obj_bound)
            
            # Those variables not added by gurobi due to range constraints
            for var in itertools.islice(pvars,self._last_native_var_idx+1):
                soln_variables[ var.VarName ] = {"Value" : var.X, "Id" : len(soln_variables) - 1}

            if extract_reduced_costs is True:
                for var in itertools.islice(pvars,self._last_native_var_idx+1):
                    soln_variables[ var.VarName ]["Rc"] = var.Rc

            if extract_duals or extract_slacks:
                for con in cons:
                    soln_constraints[con.ConstrName] = {"Id" : len(soln_constraints)}
                for con in qcons:
                    soln_constraints[con.QCName] = {"Id" : len(soln_constraints)}

            if extract_duals is True:
                for con in cons:
                    # Pi attributes in Gurobi are the constraint duals
                    soln_constraints[ con.ConstrName ]["Dual"] = con.Pi
                for con in qcons:
                    # QCPI attributes in Gurobi are the constraint duals
                    soln_constraints[ con.QCName ]["Dual"] = con.QCPi

            if extract_slacks is True:
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
                # Therefore we need to check the value of the associated slack variable
                # with its upper bound to compute the original constraint slacks. To conform
                # with the other problem writers we return the slack value that is largest in 
                # magnitude (L-f(x) or U-f(x))
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
            referenced_varnames = set(byObject[varid] for varid in self._referenced_variable_ids)
            names_to_delete = set(soln_variables.keys())-referenced_varnames
            for varname in names_to_delete:
                del soln_variables[varname]

        results.solution.insert(soln)

        self.results = results

        # Done with the model object; free up some memory.
        self._last_native_var_idx = -1
        self._range_con_var_pairs = []

        # let the base class deal with returning results.
        return OptSolver._postsolve(self)


if not gurobi_python_api_exists:
    SolverFactory().deactivate('_gurobi_direct')
    SolverFactory().deactivate('_mock_gurobi_direct')
