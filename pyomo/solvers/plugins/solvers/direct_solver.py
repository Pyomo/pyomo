from pyomo.core.base.PyomoModel import Model
from pyomo.core.kernel.component_block import IBlockStorage
from pyomo.opt.base.solvers import OptSolver
from pyomo.core.base import SymbolMap, NumericLabeler, TextLabeler
import pyutilib.common
import pyomo.opt.base.solvers
from pyomo.core.kernel.component_map import ComponentMap


class DirectSolver(OptSolver):
    """
    Subclasses need to:
    1.) Initialize self._solver_model during _presolve before calling DirectSolver._presolve
    """
    def __init__(self, **kwds):
        super(DirectSolver, self).__init__(**kwds)

        self._pyomo_model = None
        self._solver_model = None
        self._symbol_map = SymbolMap()
        self._labeler = None
        self._pyomo_var_to_solver_var_map = ComponentMap()
        self._pyomo_con_to_solver_con_map = ComponentMap()
        self._objective_label = None
        self.results = None
        self._smap_id = None
        self._skip_trivial_constraints = False
        self._output_fixed_variable_bounds = False
        self._python_api_exists = False
        self._version = None
        self._version_major = None
        self._symbolic_solver_labels = False

        self._referenced_variables = ComponentMap()
        """dict: {var: count} where count is the number of constraints/objective referencing the var"""

        # this interface doesn't use files, but we can create a log file if requested
        self._keepfiles = False

    def _presolve(self, *args, **kwds):
        """
        kwds not consumed here or at the beginning of OptSolver._presolve will raise an error in
        OptSolver._presolve.

        args
        ----
        pyomo Model or IBlockStorage

        kwds
        ----
        warmstart: bool
            can only be True if the subclass is warmstart capable; if not, an error will be raised
        symbolic_solver_labels: bool
            if True, the model will be translated using the names from the pyomo model; otherwise, the variables and
            constraints will be numbered with a generic xi
        skip_trivial_constraints: bool
            if True, any trivial constraints (e.g., 1 == 1) will be skipped (i.e., not passed to the solver).
        output_fixed_variable_bounds: bool
            if False, an error will be raised if a fixed variable is used in any expression rather than the value of the
            fixed variable.
        keepfiles: bool
            if True, the solver log file will be saved and the name of the file will be printed.

        kwds accepted by OptSolver._presolve
        """
        model = args[0]
        if len(args) != 1:
            msg = ("The {0} plugin method '_presolve' must be supplied a single problem instance - {1} were " +
                   "supplied.").format(type(self), len(args))
            raise ValueError(msg)

        self._compile_instance(model, **kwds)

        warmstart_flag = kwds.pop('warmstart', False)
        self._keepfiles = kwds.pop('keepfiles', False)

        # create a context in the temporary file manager for
        # this plugin - is "pop"ed in the _postsolve method.
        pyutilib.services.TempfileManager.push()

        self.results = None

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
        super(DirectSolver, self)._presolve(*args, **kwds)

        if warmstart_flag:
            if self.warm_start_capable():
                self._warm_start()
            else:
                raise ValueError('{0} solver plugin is not capable of warmstart.'.format(type(self)))

        if self._log_file is None:
            self._log_file = pyutilib.services.TempfileManager.create_tempfile(suffix='.log')

    def _apply_solver(self):
        raise NotImplementedError('The specific direct/persistent solver interface should implement this method.')

    def _postsolve(self):
        return super(DirectSolver, self)._postsolve()

    def _compile_instance(self, model, **kwds):
        if not isinstance(model, (Model, IBlockStorage)):
            msg = "The problem instance supplied to the {0} plugin " \
                  "'_presolve' method must be of type 'Model'".format(type(self))
            raise ValueError(msg)
        self._pyomo_model = model
        self._symbolic_solver_labels = kwds.pop('symbolic_solver_labels', self._symbolic_solver_labels)
        self._skip_trivial_constraints = kwds.pop('skip_trivial_constraints', self._skip_trivial_constraints)
        self._output_fixed_variable_bounds = kwds.pop('output_fixed_variable_bounds', self._output_fixed_variable_bounds)
        self._pyomo_var_to_solver_var_map = ComponentMap()
        self._pyomo_con_to_solver_con_map = ComponentMap()
        self._referenced_variables = ComponentMap()
        self._objective_label = None

        self._symbol_map = SymbolMap()

        if symbolic_solver_labels:
            self._labeler = TextLabeler()
        else:
            self._labeler = NumericLabeler('x')

    def _add_block(self, block):
        raise NotImplementedError('The specific direct/persistent solver interface should implement this method.')

    def _compile_objective(self):
        raise NotImplementedError('The specific direct/persistent solver interface should implement this method.')

    def _add_constraint(self, con):
        raise NotImplementedError('The specific direct/persistent solver interface should implement this method.')

    def _add_var(self, var):
        raise NotImplementedError('The specific direct/persistent solver interface should implement this method.')

    def _get_expr_from_pyomo_expr(self, expr):
        raise NotImplementedError('The specific direct/persistent solver interface should implement this method.')

    def _load_results(self):
        raise NotImplementedError('The specific direct/persistent solver interface should implement this method.')

    def warm_start_capable(self):
        return False

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
