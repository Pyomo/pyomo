#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

# NOTE: this solver is disabled (see the first try block below).  This
# code is out of date, and this code is not regularly tested by Pyomo developers.
# The python-glpk package is only supported on Debian Linux platforms, so
# it is not clear if this is a valuable solver interface, particularly since
# commercial vendors now have good support for Python interfaces (e.g. CPLEX and
# Gurobi).

import sys
_glpk_version = None
try:
    # import all the glp_* functions
    if False:  # DISABLED
        from glpk import *
        glpk_python_api_exists = True
    else:
        glpk_python_api_exists = False
except ImportError:
    glpk_python_api_exists = False
except Exception as e:
    # other forms of exceptions can be thrown by the glpk python
    # import. For example, an error in code invoked by the module's
    # __init__.  We should continue gracefully and not cause a fatal
    # error in Pyomo.
    print("Import of glpk failed - glpk message="+str(e)+"\n")
    glpk_python_api_exists = False

from pyutilib.misc import Bunch, Options

from pyomo.util.plugin import alias
from pyomo.opt.base import *
from pyomo.opt.base.solvers import _extract_version
from pyomo.opt.results import *
from pyomo.opt.solver import *
from pyomo.core.base.numvalue import value

import logging
logger = logging.getLogger('pyomo.solvers')



class GLPKDirect ( OptSolver ):
    """The GLPK LP/MIP solver (direct API plugin)

 The glpk_direct plugin offers an API interface to the GLPK.  It requires the
 Python-GLPK API interface provided via the SWIG interface available through
 most Linux distributions repositories.  For more information, see Jo~ao Pedro
 Pedroso's (the author) page at http://www.dcc.fc.up.pt/~jpp/

 Because of the direct connection with the GLPK API, no temporary files need be
 written or read.  That ostensibly makes this a faster plugin than the
 file-based glpk plugin.  However, you will likely not notice any speed up
 unless you are using the GLPK solver with PySP problems (due to the rapid
 re-solves).

 One downside to the lack of temporary files, is that there is no LP file to
 inspect for clues while debugging a model.  For that, use the write_lp solver
 option:

 $ pyomo model.{py,dat} \
   --solver=glpk_direct \
   --solver-options  write_lp=/path/to/some/file.lp

 One can also specify the particular GLPK algorithm to use with the 'algorithm'
 solver option.  There are 4 algorithms:

   simplex  - the default algorithm for non-MIP problems (primal simplex)
   intopt   - the default algorithm for MIP problems
   exact    - tentative implementation of two-phase primal simplex with exact
              arithmetic internally.
   interior - only aplicable to LP problems

 $ pyomo model.{py,dat} \
   --solver glpk_direct \
   --solver-options  algorithm=exact

 For more information on available algorithms, see the GLPK documentation.
    """

    alias('_glpk_direct', doc='Direct Python interface to the GLPK LP/MIP solver.')


    def __init__(self, **kwds):
        #
        # Call base class constructor
        #
        kwds['type'] = 'glpk_direct'
        OptSolver.__init__(self, **kwds)

        # NOTE: eventually both of the following attributes should be migrated
        # to a common base class.  Is the current solve warm-started?  A
        # transient data member to communicate state information across the
        # _presolve, _apply_solver, and _postsolve methods.
        self.warm_start_solve = False
        self._timelimit = None

        # Note: Undefined capabilities default to 'None'
        self._capabilities = Options()
        self._capabilities.linear = True
        self._capabilities.integer = True

    def _get_version(self):
        """
        Returns a tuple describing the solver executable version.
        """
        if _glpk_version is None:
            return _extract_version('')
        return _glpk_version

    def _populate_glpk_instance ( self, model ):

        from pyomo.core.base import Var, Objective, Constraint, SOSConstraint

        try:
            lp = glp_create_prob()
        except Exception:
            e = sys.exc_info()[1]
            msg = 'Unable to create GLPK problem instance.  Have you installed' \
            '\n       the Python bindings for GLPK?\n\n\tError message: %s'
            raise Exception(msg % e)

        objective = sorted( model.component_map(Objective, active=True).values() )[0]
        # so we can correctly map the solution to the correct objective label in _postsolve
        lp.objective_name = sorted( model.component_map(Objective, active=True).keys() )[0]
        sense = GLP_MAX
        if objective.is_minimizing(): sense = GLP_MIN

        constraint_list = model.component_map(Constraint, active=True)
        variable_list   = model.component_map(Var, active=True)
        num_constraints = model.statistics.number_of_constraints
        num_variables   = model.statistics.number_of_variables

        sosn = self._capabilities.sosn
        sos1 = self._capabilities.sos1
        sos2 = self._capabilities.sos2

        for soscondata in model.component_data_objects(SOSConstraint, active=True):
            raise Exception("Solver: glpk_direct does not support SOSConstraint declarations")

        glp_set_prob_name(lp, model.name)

        glp_set_obj_dir( lp, sense )
        glp_add_rows( lp, num_constraints )
        glp_add_cols( lp, num_variables )

        # 1 extra because GLPK's arrays in this context are 1-based, not 0-based
        coef_count = num_constraints * num_variables + 1
        Ai = intArray( coef_count )
        Aj = intArray( coef_count )
        Ar = doubleArray( coef_count )

        row = col = coef_count = 0
        colvar_map = dict()
        rowvar_map = dict()

        # In matrix parlance, variables are columns
        for name in variable_list:
            var_set = variable_list[ name ]
            for ii in var_set:
                var = var_set[ ii ]
                if var.fixed is True:
                    continue

                lb = ub = 0.0
                if var.lb is None and var.ub is None:
                    var_type = GLP_FR
                elif var.lb is None:
                    var_type = GLP_UB
                    ub = value(var.ub)
                elif var.ub is None:
                    var_type = GLP_LO
                    lb = value(var.lb)
                else:
                    var_type = GLP_DB
                    lb = value(var.lb)
                    ub = value(var.ub)

                col += 1
                colvar_map[ var.label ] = col

                # the name is perhaps not necessary, but for completeness ...
                glp_set_col_name( lp, col, var.label )
                glp_set_col_bnds( lp, col, var_type, lb, ub )

                # Be sure to impart the integer and binary nature of any variables
                if var.is_integer():
                    glp_set_col_kind( lp, col, GLP_IV )
                elif var.is_binary():
                    glp_set_col_kind( lp, col, GLP_BV )
                elif var.is_continuous():
                    glp_set_col_kind( lp, col, GLP_CV )   # continuous
                else:
                    raise TypeError("Invalid domain type for variable with name '%s'. "
                                    "Variable is not continuous, integer, or binary.")

        model_canonical_repn = getattr(model, "_canonical_repn", None)
        if model_canonical_repn is None:
            raise ValueError("No _canonical_repn ComponentMap was found on "
                             "block with name %s. Did you forget to preprocess?"
                             % (model.name))

        for name in constraint_list:
            constraint_set = constraint_list[ name ]

            for ii in constraint_set:
                constraint = constraint_set[ ii ]
                if not constraint.active: continue
                elif constraint.lower is None and constraint.upper is None:
                    continue

                expression = model_canonical_repn.get(constraint)
                if constraint is None:
                    raise ValueError("No entry found in _canonical_repn ComponentMap on "
                                     "block %s for active constraint with name %s. "
                                     "Did you forget to preprocess?"
                                     % (model.name, constraint.name))

                offset = 0.0
                if 0 in expression:
                    offset = expression[0][None]

                lbound = ubound = -offset

                if constraint.equality:
                    var_type = GLP_FX    # Fixed
                    lbound = ubound = constraint.lower() - offset
                elif constraint.lower is None:
                    var_type = GLP_UP    # Upper bounded only
                    ubound += constraint.upper()
                elif constraint.upper is None:
                    var_type = GLP_LO    # Lower bounded only
                    lbound += constraint.lower()
                else:
                    var_type = GLP_DB    # Double bounded
                    lbound += constraint.lower()
                    ubound += constraint.upper()

                row += 1
                rowvar_map[ constraint.label ] = row

                # just as with variables, set the name just for completeness ...
                glp_set_row_name( lp, row, constraint.label )
                glp_set_row_bnds( lp, row, var_type, lbound, ubound )

                if 1 in expression: # first-order terms
                    keys = sorted( expression[1].keys() )
                    for var_key in keys:
                        index = var_key.keys()[0]
                        var = expression[-1][ index ]
                        coef  = expression[ 1][ var_key ]
                        col = colvar_map[ var.label ]

                        coef_count += 1
                        Ai[ coef_count ] = row
                        Aj[ coef_count ] = col
                        Ar[ coef_count ] = coef

        # with the rows and columns named and bounded, load the coefficients
        glp_load_matrix( lp, coef_count, Ai, Aj, Ar )

        for key in objective:

            expression = model_canonical_repn.get(objective[key])
            if expression is None:
                raise ValueError("No entry found in _canonical_repn ComponentMap on "
                                 "block %s for active objective with name %s. "
                                 "Did you forget to preprocess?"
                                 % (model.name, objective[key].name))

            if expression.is_constant():
                msg = "Ignoring objective '%s[%s]' which is constant"
                logger.warning( msg % (str(objective), str(key)) )
                continue

            if 1 in expression: # first-order terms
                keys = sorted( expression[1].keys() )
                for var_key in keys:
                    index = var_key.keys()[0]
                    label = expression[-1][ index ].label
                    coef  = expression[ 1][ var_key ]
                    col = colvar_map[ label ]
                    glp_set_obj_coef( lp, col, coef )

            elif -1 in expression:
                pass
            else:
                msg = "Nonlinear objective to GLPK.  GLPK can only handle "       \
                      "linear problems."
                raise RuntimeError( msg )


        self._glpk_instance = lp
        self._glpk_rowvar_map = rowvar_map
        self._glpk_colvar_map = colvar_map


    def warm_start_capable(self):
        # Note to dev who gets back here: GLPK has the notion of presolving, but
        # it's "built-in" to each optimization function.  To disable it, make use
        # of the second argument of glp_smcp type.  See the GLPK documentation
        # PDF for further information.
        msg = "GLPK has the ability to use warmstart solutions.  However, it "  \
              "has not yet been implemented into the Pyomo glpk_direct plugin."
        logger.info( msg )
        return False


    def warm_start(self, instance):
        pass


    def _presolve(self, *args, **kwargs):
        from pyomo.core.base.PyomoModel import Model

        self.warm_start_solve = kwargs.pop( 'warmstart', False )

        model = args[0]
        if len(args) != 1:
            msg = "The glpk_direct plugin method '_presolve' must be supplied "  \
                  "a single problem instance - %s were supplied"
            raise ValueError(msg % len(args))
        elif not isinstance(model, Model):
            msg = "The problem instance supplied to the glpk_direct plugin "     \
                  "'_presolve' method must be of type 'Model'"
            raise ValueError(msg)

        self._populate_glpk_instance( model )
        lp = self._glpk_instance
        self.is_integer = ( glp_get_num_int( lp ) > 0 and True or False )

        if 'write_lp' in self.options:
            fname = self.options.write_lp
            glp_write_lp( lp, None, fname )

        self.algo = 'Simplex (primal)'
        algorithm = glp_simplex
        if self.is_integer > 0:
            self.algo = 'Mixed Integer'
            algorithm = glp_intopt

        if 'algorithm' in self.options:
            if 'simplex' == self.options.algorithm:
                self.algo = 'Simplex (primal)'
                algorithm = glp_simplex
            elif 'exact' == self.options.algorithm:
                self.algo = 'Simplex (two-phase primal)'
                algorithm = glp_exact
            elif 'interior' == self.options.algorithm:
                self.algo = 'Interior Point'
                algorithm = glp_interior
            elif 'intopt' == self.options.algorithm:
                self.alpo = 'Mixed Integer'
                algorithm = glp_intopt
            else:
                msg = "Unknown solver specified\n  Unknown: %s\n  Using:   %s\n"
                logger.warning( msg % (self.options.algorithm, self.algo) )
        self._algorithm = algorithm

        if 'Simplex (primal)' == self.algo:
            parm = glp_smcp()
            glp_init_smcp( parm )
        elif 'Simplex (two-phase primal)' == self.algo:
            parm = glp_smcp()
            glp_init_smcp( parm )
        elif 'Interior Point' == self.algo:
            parm = glp_iptcp()
            glp_init_iptcp( parm )
        elif 'Mixed Integer' == self.algo:
            parm = glp_iocp()
            glp_init_iocp( parm )
            if self.options.mipgap:
                parm.mip_gap = self.options.mipgap

        if self._timelimit and self._timelimit > 0.0:
            parm.tm_lim = self._timelimit

        parm.msg_lev = GLP_MSG_OFF
        parm.presolve = GLP_ON

        self._solver_params = parm

        # Scaffolding in place: I /believe/ GLPK can do warmstarts; just supply
        # the basis solution and turn of the presolver
        if self.warm_start_solve is True:

            if len(args) != 1:
                msg = "The glpk_direct _presolve method can only handle a single "\
                      "problem instance - %s were supplied"
                raise ValueError(msg % len(args))

            parm.presolve = GLP_OFF
            self.warm_start( model )


    def _apply_solver(self):
        lp = self._glpk_instance
        parm = self._solver_params
        algorithm = self._algorithm

        # Actually solve the problem.
        try:
            beg = glp_time()
            self.solve_return_code = algorithm( self._glpk_instance, parm )
            end = glp_time()
            self._glpk_solve_time = glp_difftime( end, beg )
        except Exception:
            e = sys.exc_info()[1]
            msg = str(e)
            if 'algorithm' in self.options:
                msg = "Unexpected error using '%s' algorithm.  Is it the correct "\
                      "correct algorithm for the problem type?"
                msg %= self.options.algorithm
            logger.error( msg )
            raise

        # FIXME: can we get a return code indicating if GLPK had a
        # significant failure?
        return Bunch(rc=None, log=None)


    def _glpk_return_code_to_message ( self ):
        code = self.solve_return_code
        if 0 == code:
            return "Algorithm completed successfully.  (This does not "         \
                   "necessarily mean an optimal solution was found.)"
        elif GLP_EBADB == code:
            return "Unable to start the search, because the initial basis "     \
               "specified in the problem object is invalid -- the number of "   \
               "basic (auxiliary and structural) variables is not the same as " \
               "the number of rows in the problem object."
        elif GLP_ESING == code:
            return "Unable to start the search, because the basis matrix "      \
               "corresponding to the initial basis is singular within the "     \
               "working precision."
        elif GLP_ECOND == code:
            return "Unable to start the search, because the basis matrix "       \
               "corresponding to the initial basis is ill-conditioned, i.e. its "\
               "condition number is too large."
        elif GLP_EBOUND == code:
            return "Unable to start the search, because some double-bounded "    \
               "(auxiliary or structural) variables have incorrect bounds."
        elif GLP_EFAIL == code:
            return "The search was prematurely terminated due to the solver "    \
               "failure."
        elif GLP_EOBJLL == code:
            return "The search was prematurely terminated, because the "         \
               "objective function being maximized has reached its lower limit " \
               "and continues decreasing (the dual simplex only)."
        elif GLP_EOBJUL == code:
            return "The search was prematurely terminated, because the "         \
               "objective function being minimized has reached its upper limit " \
               "and continues increasing (the dual simplex only)."
        elif GLP_EITLIM == code:
            return "The search was prematurely terminated, because the simplex " \
               "iteration limit has been exceeded."
        elif GLP_ETMLIM == code:
            return "The search was prematurely terminated, because the time "    \
               "limit has been exceeded."
        elif GLP_ENOPFS == code:
            return "The LP problem instance has no primal feasible solution "    \
               "(only if the LP presolver is used)."
        elif GLP_ENODFS == code:
            return "The LP problem instance has no dual feasible solution "     \
               "(only if the LP presolver is used)."
        else:
            return "Unexpected error condition.  Please consider remitting "    \
               "this problem to the Pyomo developers and/or the GLPK project "  \
               "so they can improve their softwares."


    def _glpk_get_solution_status ( self ):
        getstatus = glp_get_status
        if self.is_integer: getstatus = glp_mip_status

        status = getstatus( self._glpk_instance )
        if   GLP_OPT    == status: return SolutionStatus.optimal
        elif GLP_FEAS   == status: return SolutionStatus.feasible
        elif GLP_INFEAS == status: return SolutionStatus.infeasible
        elif GLP_NOFEAS == status: return SolutionStatus.other
        elif GLP_UNBND  == status: return SolutionStatus.other
        elif GLP_UNDEF  == status: return SolutionStatus.other
        raise RuntimeError("Unknown solution status returned by GLPK solver")


    def _glpk_get_solver_status ( self ):
        rc = self.solve_return_code
        if 0 == rc:            return SolverStatus.ok
        elif GLP_EBADB  == rc: return SolverStatus.error
        elif GLP_ESING  == rc: return SolverStatus.error
        elif GLP_ECOND  == rc: return SolverStatus.error
        elif GLP_EBOUND == rc: return SolverStatus.error
        elif GLP_EFAIL  == rc: return SolverStatus.aborted
        elif GLP_EOBJLL == rc: return SolverStatus.aborted
        elif GLP_EOBJUL == rc: return SolverStatus.aborted
        elif GLP_EITLIM == rc: return SolverStatus.aborted
        elif GLP_ETMLIM == rc: return SolverStatus.aborted
        elif GLP_ENOPFS == rc: return SolverStatus.warning
        elif GLP_ENODFS == rc: return SolverStatus.warning
        else: return SolverStatus.unkown


    def _postsolve(self):
        lp = self._glpk_instance
        num_variables = glp_get_num_cols( lp )
        bin_variables = glp_get_num_bin( lp )
        int_variables = glp_get_num_int( lp )

        # check suffixes
        for suffix in self._suffixes:
            if True:
                raise RuntimeError("***The glpk_direct solver plugin cannot extract solution suffix="+suffix)


        tpeak = glp_long()
        glp_mem_usage( None, None, None, tpeak )
        # black magic trickery, thanks to Python's lack of pointers and SWIG's
        # automatic API conversion
        peak_mem = tpeak.lo

        results = SolverResults()
        soln = Solution()
        prob = results.problem
        solv = results.solver

        solv.name = "GLPK " + glp_version()
        solv.status = self._glpk_get_solver_status()
        solv.return_code = self.solve_return_code
        solv.message = self._glpk_return_code_to_message()
        solv.algorithm = self.algo
        solv.memory_used = "%d bytes, (%d KiB)" % (peak_mem, peak_mem/1024)
        # solv.user_time = None
        # solv.system_time = None
        solv.wallclock_time = self._glpk_solve_time
        # solv.termination_condition = None
        # solv.termination_message = None

        prob.name = glp_get_prob_name(lp)
        prob.number_of_constraints = glp_get_num_rows(lp)
        prob.number_of_nonzeros = glp_get_num_nz(lp)
        prob.number_of_variables = num_variables
        prob.number_of_binary_variables = bin_variables
        prob.number_of_integer_variables = int_variables
        prob.number_of_continuous_variables = num_variables - int_variables
        prob.number_of_objectives = 1

        prob.sense = ProblemSense.minimize
        if GLP_MAX == glp_get_obj_dir( lp ):
            prob.sense = ProblemSense.maximize

        soln.status = self._glpk_get_solution_status()

        if soln.status in ( SolutionStatus.optimal, SolutionStatus.feasible ):
            get_col_prim = glp_get_col_prim
            get_row_prim = glp_get_row_prim
            get_obj_val  = glp_get_obj_val
            if self.is_integer:
                get_col_prim = glp_mip_col_val
                get_row_prim = glp_mip_row_val
                get_obj_val  = glp_mip_obj_val

            obj_val = get_obj_val( lp )
            if prob.sense == ProblemSense.minimize:
                prob.lower_bound = obj_val
            else:
                prob.upper_bound = obj_val

            objective_name = lp.objective_name
            soln.objective[objective_name] = {'Value': obj_val}

            colvar_map = self._glpk_colvar_map
            rowvar_map = self._glpk_rowvar_map

            for var_label in colvar_map:
                col = colvar_map[ var_label ]
                soln.variable[ var_label ] = {"Value" : get_col_prim( lp, col )}

            for row_label in rowvar_map:
                row = rowvar_map[ row_label ]
                soln.constraint[ row_label ] = {"Value" : get_row_prim( lp, row )}

        results.solution.insert(soln)

        self.results = results

        # All done with the GLPK object, so free up some memory.
        glp_free( lp )
        del self._glpk_instance, lp

        # let the base class deal with returning results.
        return OptSolver._postsolve(self)


# TODO: add MockGLPKDirect class

if not glpk_python_api_exists:
    SolverFactory().deactivate('_glpk_direct')
    # SolverFactory().deactivate('_mock_glpk_direct')

# vim: set fileencoding=utf-8
