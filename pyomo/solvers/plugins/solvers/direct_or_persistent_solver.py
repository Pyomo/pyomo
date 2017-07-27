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
from pyomo.core.kernel.component_block import IBlockStorage
from pyomo.opt.base.solvers import OptSolver
from pyomo.core.base import SymbolMap, NumericLabeler, TextLabeler
import pyutilib.common
import pyutilib.services
import pyomo.opt.base.solvers
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.opt.base.formats import ResultsFormat
from pyutilib.misc import Options


class DirectOrPersistentSolver(OptSolver):
    """
    Subclasses need to:
    1.) Initialize self._solver_model during _presolve before calling DirectSolver._presolve
    """
    def __init__(self, **kwds):
        OptSolver.__init__(self, **kwds)

        self._pyomo_model = None
        self._solver_model = None
        self._symbol_map = SymbolMap()
        self._labeler = None
        self._pyomo_var_to_solver_var_map = ComponentMap()
        self._pyomo_con_to_solver_con_map = ComponentMap()
        self._vars_referenced_by_con = ComponentMap()
        self._vars_referenced_by_obj = ComponentSet()
        self._objective_label = None
        self._objective = None
        self.results = None
        self._smap_id = None
        self._skip_trivial_constraints = False
        self._output_fixed_variable_bounds = False
        self._python_api_exists = False
        self._version = None
        self._version_major = None
        self._symbolic_solver_labels = False
        self._capabilites = Options()

        self._referenced_variables = ComponentMap()
        """dict: {var: count} where count is the number of constraints/objective referencing the var"""

        # this interface doesn't use files, but we can create a log file if requested
        self._keepfiles = False

    def _presolve(self, *args, **kwds):
        warmstart_flag = kwds.pop('warmstart', False)
        self._keepfiles = kwds.pop('keepfiles', False)

        # create a context in the temporary file manager for
        # this plugin - is "pop"ed in the _postsolve method.
        pyutilib.services.TempfileManager.push()

        self.results = None

        model = self._pyomo_model
        self._smap_id = id(self._symbol_map)
        if isinstance(model, IBlockStorage):
            # BIG HACK (see pyomo.core.kernel write function)
            if not hasattr(model, "._symbol_maps"):
                setattr(model, "._symbol_maps", {})
            getattr(model, "._symbol_maps")[self._smap_id] = self._symbol_map
        else:
            model.solutions.add_symbol_map(self._symbol_map)

        # this implies we have a custom solution "parser",
        # preventing the OptSolver _presolve method from
        # creating one
        self._results_format = ResultsFormat.soln
        # use the base class _presolve to consume the
        # important keywords
        OptSolver._presolve(self, *args, **kwds)

        if warmstart_flag:
            if self.warm_start_capable():
                self._warm_start()
            else:
                raise ValueError('{0} solver plugin is not capable of warmstart.'.format(type(self)))

        if self._log_file is None:
            self._log_file = pyutilib.services.TempfileManager.create_tempfile(suffix='.log')

    """ This method should be implemented by subclasses."""
    def _apply_solver(self):
        raise NotImplementedError('This method should be implemented by subclasses')

    """ This method should be implemented by subclasses."""
    def _postsolve(self):
        return OptSolver._postsolve(self)

    """ This method should be implemented by subclasses."""
    def _compile_instance(self, model, kwds={}):
        if not isinstance(model, (Model, IBlockStorage)):
            msg = "The problem instance supplied to the {0} plugin " \
                  "'_presolve' method must be of type 'Model'".format(type(self))
            raise ValueError(msg)
        self._pyomo_model = model
        self._symbolic_solver_labels = kwds.pop('symbolic_solver_labels', self._symbolic_solver_labels)
        self._skip_trivial_constraints = kwds.pop('skip_trivial_constraints', self._skip_trivial_constraints)
        self._output_fixed_variable_bounds = kwds.pop('output_fixed_variable_bounds',
                                                      self._output_fixed_variable_bounds)
        self._pyomo_var_to_solver_var_map = ComponentMap()
        self._pyomo_con_to_solver_con_map = ComponentMap()
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
        for var in block.component_data_objects(ctype=pyomo.core.base.var.Var, descend_into=True,
                                                active=True, sort=True):
            self._add_var(var)

        for sub_block in block.block_data_objects(descend_into=True, active=True):
            for con in sub_block.component_data_objects(ctype=pyomo.core.base.constraint.Constraint,
                                                        descend_into=False, active=True, sort=True):
                self._add_constraint(con)

            for con in sub_block.component_data_objects(ctype=pyomo.core.base.sos.SOSConstraint,
                                                        descend_into=False, active=True, sort=True):
                self._add_sos_constraint(con)

            if len([obj for obj in sub_block.component_data_objects(ctype=pyomo.core.base.objective.Objective,
                                                                    descend_into=False, active=True)]) != 0:
                self._compile_objective()

    """ This method should be implemented by subclasses."""
    def _compile_objective(self):
        raise NotImplementedError('This method should be implemented by subclasses')

    """ This method should be implemented by subclasses."""
    def _add_constraint(self, con):
        raise NotImplementedError('This method should be implemented by subclasses')

    """ This method should be implemented by subclasses."""
    def _add_sos_constraint(self, con):
        raise NotImplementedError('This method should be implemented by subclasses')

    """ This method should be implemented by subclasses."""
    def _add_var(self, var):
        raise NotImplementedError('This method should be implemented by subclasses')

    """ This method should be implemented by subclasses."""
    def _get_expr_from_pyomo_repn(self, repn, max_degree=None):
        raise NotImplementedError('This method should be implemented by subclasses')

    """ This method should be implemented by subclasses."""
    def _get_expr_from_pyomo_expr(self, expr, max_degree=None):
        raise NotImplementedError('This method should be implemented by subclasses')

    """ This method should be implemented by subclasses."""
    def _load_vars(self, vars_to_load):
        raise NotImplementedError('This method should be implemented by subclasses')

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
                raise pyutilib.common.ApplicationError(("No Python bindings available for {0} solver " +
                                                        "plugin").format(type(self)))
            else:
                return True

    def _get_version(self):
        if self._version is None:
            return pyomo.opt.base.solvers._extract_version('')
        return self._version
