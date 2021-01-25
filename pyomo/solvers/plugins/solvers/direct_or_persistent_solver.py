#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.base.PyomoModel import Model
from pyomo.core.base.block import Block, _BlockData
from pyomo.core.kernel.block import IBlock
from pyomo.opt.base.solvers import OptSolver
from pyomo.core.base import SymbolMap, NumericLabeler, TextLabeler
import pyomo.common
from pyomo.common.errors import ApplicationError
from pyomo.common.collections import ComponentMap, ComponentSet, Options
from pyomo.common.tempfiles import TempfileManager
import pyomo.opt.base.solvers
from pyomo.opt.base.formats import ResultsFormat


class DirectOrPersistentSolver(OptSolver):
    """
    This is a base class for both direct and persistent solvers. Direct solver interfaces do not use any file io.
    Rather, they interface directly with the python bindings for the specific solver. Persistent solver interfaces
    are similar except that they "remember" their model. Thus, persistent solver interfaces allow incremental changes
    to the solver model (e.g., the gurobi python model or the cplex python model). Note that users are responsible
    for notifying the persistent solver interfaces when changes are made to the corresponding pyomo model.

    Parameters
    ----------
    type: str
        String indicating the class type of the solver instance.
    name: str
        String representing either the class type of the solver instance or an assigned name.
    doc: str
        Documentation for the solver
    options: dict
        Dictionary of solver options
    """
    def __init__(self, **kwds):
        OptSolver.__init__(self, **kwds)

        self._pyomo_model = None
        """The pyomo model being solved."""

        self._solver_model = None
        """The python instance of the solver model (e.g., the gurobipy Model instance)."""

        self._symbol_map = SymbolMap()
        """A symbol map used to map between pyomo components and their names used with the solver."""

        self._labeler = None
        """The labeler for creating names for the solver model components."""

        self._pyomo_var_to_solver_var_map = ComponentMap()
        self._solver_var_to_pyomo_var_map = dict()
        """A dictionary mapping pyomo Var's to the solver variables."""

        self._pyomo_con_to_solver_con_map = dict()
        self._solver_con_to_pyomo_con_map = dict()
        """A dictionary mapping pyomo constraints to solver constraints."""

        self._vars_referenced_by_con = ComponentMap()
        """A dictionary mapping constraints to a ComponentSet containt the pyomo variables referenced by that
        constraint. This is primarily needed for the persistent solvers. When a constraint is deleted, we need
        to decrement the number of times those variables are referenced (see self._referenced_variables)."""

        self._vars_referenced_by_obj = ComponentSet()
        """A set containing the pyomo variables referenced by that the objective.
        This is primarily needed for the persistent solvers. When a the objective is deleted, we need
        to decrement the number of times those variables are referenced (see self._referenced_variables)."""

        self._objective = None
        """The pyomo Objective object currently being used with the solver."""

        self.results = None
        """A results object return from the solve method."""

        self._skip_trivial_constraints = False
        """A bool. If True, then any constraints with a constant body will not be added to the solver model.
        Be careful with this. If a trivial constraint is skipped then that constraint cannot be removed from
        a persistent solver (an error will be raised if a user tries to remove a non-existent constraint)."""

        self._output_fixed_variable_bounds = False
        """A bool. If False then an error will be raised if a fixed variable is used in one of the solver constraints.
        This is useful for catching bugs. Ordinarily a fixed variable should appear as a constant value in the
        solver constraints. If True, then the error will not be raised."""

        self._python_api_exists = False
        """A bool indicating whether or not the python api is available for the specified solver."""

        self._version = None
        """The version of the solver."""

        self._version_major = None
        """The major version of the solver. For example, if using Gurobi 7.0.2, then _version_major is 7."""

        self._symbolic_solver_labels = False
        """A bool. If true then the solver components will be given names corresponding to the pyomo component names."""

        self._capabilites = Options()

        self._referenced_variables = ComponentMap()
        """dict: {var: count} where count is the number of constraints/objective referencing the var"""

        self._keepfiles = False
        """A bool. If True, then the solver log will be saved."""

        self._save_results = True
        """A bool. This is used for backwards compatability. If True, the solution will be loaded into the Solution
        object that gets placed on the SolverResults object. This way, users can do model.solutions.load_from(results)
        to load solutions into thier model. However, it is more efficient to bypass the Solution object and load
        the results directly from the solver object. If False, the solution will not be loaded into the Solution
        object."""

    def _presolve(self, **kwds):
        warmstart_flag = kwds.pop('warmstart', False)
        self._keepfiles = kwds.pop('keepfiles', False)
        self._save_results = kwds.pop('save_results', True)

        # create a context in the temporary file manager for
        # this plugin - is "pop"ed in the _postsolve method.
        TempfileManager.push()

        self.results = None

        model = self._pyomo_model

        # this implies we have a custom solution "parser",
        # preventing the OptSolver _presolve method from
        # creating one
        self._results_format = ResultsFormat.soln
        # use the base class _presolve to consume the
        # important keywords
        OptSolver._presolve(self, **kwds)

        # ***********************************************************
        # The following code is only needed for backwards compatability of load_solutions=False.
        # If we ever only want to support the load_vars, load_duals, etc. methods, then this can be deleted.
        if self._save_results:
            self._smap_id = id(self._symbol_map)
            if isinstance(self._pyomo_model, IBlock):
                # BIG HACK (see pyomo.core.kernel write function)
                if not hasattr(self._pyomo_model, "._symbol_maps"):
                    setattr(self._pyomo_model, "._symbol_maps", {})
                getattr(self._pyomo_model,
                        "._symbol_maps")[self._smap_id] = self._symbol_map
            else:
                self._pyomo_model.solutions.add_symbol_map(self._symbol_map)
        # ***********************************************************

        if warmstart_flag:
            if self.warm_start_capable():
                self._warm_start()
            else:
                raise ValueError('{0} solver plugin is not capable of warmstart.'.format(type(self)))

        if self._log_file is None:
            self._log_file = TempfileManager.create_tempfile(suffix='.log')

    """ This method should be implemented by subclasses."""
    def _apply_solver(self):
        raise NotImplementedError('This method should be implemented by subclasses')

    """ This method should be implemented by subclasses."""
    def _postsolve(self):
        return OptSolver._postsolve(self)

    """ This method should be implemented by subclasses."""
    def _set_instance(self, model, kwds={}):
        if not isinstance(model, (Model, IBlock, Block, _BlockData)):
            msg = "The problem instance supplied to the {0} plugin " \
                  "'_presolve' method must be a Model or a Block".format(type(self))
            raise ValueError(msg)
        self._pyomo_model = model
        self._symbolic_solver_labels = kwds.pop('symbolic_solver_labels', self._symbolic_solver_labels)
        self._skip_trivial_constraints = kwds.pop('skip_trivial_constraints', self._skip_trivial_constraints)
        self._output_fixed_variable_bounds = kwds.pop('output_fixed_variable_bounds',
                                                      self._output_fixed_variable_bounds)
        self._pyomo_var_to_solver_var_map = ComponentMap()
        self._solver_var_to_pyomo_var_map = dict()
        self._pyomo_con_to_solver_con_map = dict()
        self._solver_con_to_pyomo_con_map = dict()
        self._vars_referenced_by_con = ComponentMap()
        self._vars_referenced_by_obj = ComponentSet()
        self._referenced_variables = ComponentMap()
        self._objective_label = None
        self._objective = None

        self._symbol_map = SymbolMap()

        if self._symbolic_solver_labels:
            self._labeler = TextLabeler()
        else:
            self._labeler = NumericLabeler('x')

    def _add_block(self, block):
        for var in block.component_data_objects(
                ctype=pyomo.core.base.var.Var,
                descend_into=True,
                active=True,
                sort=True):
            self._add_var(var)

        for sub_block in block.block_data_objects(descend_into=True,
                                                  active=True):
            for con in sub_block.component_data_objects(
                    ctype=pyomo.core.base.constraint.Constraint,
                    descend_into=False,
                    active=True,
                    sort=True):
                if (not con.has_lb()) and \
                   (not con.has_ub()):
                    assert not con.equality
                    continue  # non-binding, so skip
                self._add_constraint(con)

            for con in sub_block.component_data_objects(
                    ctype=pyomo.core.base.sos.SOSConstraint,
                    descend_into=False,
                    active=True,
                    sort=True):
                self._add_sos_constraint(con)

            obj_counter = 0
            for obj in sub_block.component_data_objects(
                    ctype=pyomo.core.base.objective.Objective,
                    descend_into=False,
                    active=True):
                obj_counter += 1
                if obj_counter > 1:
                    raise ValueError("Solver interface does not "
                                     "support multiple objectives.")
                self._set_objective(obj)

    """ This method should be implemented by subclasses."""
    def _set_objective(self, obj):
        raise NotImplementedError("This method should be implemented "
                                  "by subclasses")

    """ This method should be implemented by subclasses."""
    def _add_constraint(self, con):
        raise NotImplementedError("This method should be implemented "
                                  "by subclasses")

    """ This method should be implemented by subclasses."""
    def _add_sos_constraint(self, con):
        raise NotImplementedError("This method should be implemented "
                                  "by subclasses")

    """ This method should be implemented by subclasses."""
    def _add_var(self, var):
        raise NotImplementedError("This method should be implemented "
                                  "by subclasses")

    """ This method should be implemented by subclasses."""
    def _get_expr_from_pyomo_repn(self, repn, max_degree=None):
        raise NotImplementedError("This method should be implemented "
                                  "by subclasses")

    """ This method should be implemented by subclasses."""
    def _get_expr_from_pyomo_expr(self, expr, max_degree=None):
        raise NotImplementedError("This method should be implemented "
                                  "by subclasses")

    """ This method should be implemented by subclasses."""
    def _load_vars(self, vars_to_load):
        raise NotImplementedError("This method should be implemented "
                                  "by subclasses")

    def load_vars(self, vars_to_load=None):
        """
        Load the values from the solver's variables into the corresponding pyomo variables.

        Parameters
        ----------
        vars_to_load: list of Var
        """
        self._load_vars(vars_to_load)

    """ This method should be implemented by subclasses."""
    def warm_start_capable(self):
        raise NotImplementedError('This method should be implemented by subclasses')

    def _warm_start(self):
        raise NotImplementedError('If a subclass can warmstart, then it should implement this method.')

    def available(self, exception_flag=True):
        """True if the solver is available."""

        if exception_flag is False:
            return self._python_api_exists
        else:
            if self._python_api_exists is False:
                raise ApplicationError(("No Python bindings available for {0} solver " +
                                                        "plugin").format(type(self)))
            else:
                return True

    def _get_version(self):
        if self._version is None:
            return pyomo.opt.base.solvers._extract_version('')
        return self._version
