from pyomo.core.base.PyomoModel import Model
from pyomo.core.kernel.component_block import IBlockStorage
from pyomo.opt.base.solvers import OptSolver
from pyomo.core.base import SymbolMap, NumericLabeler, TextLabeler
import pyutilib.common
import pyomo.opt.base.solvers
from pyomo.core.kernel.component_map import ComponentMap


class PersistentSolver(OptSolver):

    def __init__(self, **kwds):
        super(PersistentSolver, self).__init__(**kwds)

        # Ensure any subclasses inherit from PersistentSolver before any direct solver
        assert type(self).__bases__[0] is PersistentSolver

        self._pyomo_model = kwds.pop('model', None)
        self._solver_model = None
        self._symbol_map = None
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

        if self._pyomo_model is not None:
            self.compile_instance(self._pyomo_model, **kwds)

    def _presolve(self, *args, **kwds):
        if len(args) != 0:
            msg = 'The persistent solver interface does not accept a problem instance in the solve method.'
            msg += ' The problem instance should be compiled before the solve using the compile_instance method.'
            raise ValueError(msg)

        warmstart_flag = kwds.pop('warmstart', False)
        self._keepfiles = kwds.pop('keepfiles', False)

        # create a context in the temporary file manager for
        # this plugin - is "pop"ed in the _postsolve method.
        pyutilib.services.TempfileManager.push()

        self.results = None

        # this implies we have a custom solution "parser",
        # preventing the OptSolver _presolve method from
        # creating one
        self._results_format = ResultsFormat.soln
        # use the base class _presolve to consume the
        # important keywords
        super(PersistentSolver, self)._presolve(*args, **kwds)

        if warmstart_flag:
            if self.warm_start_capable():
                self._warm_start()
            else:
                raise ValueError('{0} solver plugin is not capable of warmstart.'.format(type(self)))

        if self._log_file is None:
            self._log_file = pyutilib.services.TempfileManager.create_tempfile(suffix='.log')

    def _apply_solver(self):
        raise NotImplementedError('The subclass should implement this method.')

    def _postsolve(self):
        raise NotImplementedError('The subclass should implement this method.')

    def _compile_instance(self, model, **kwds):
        raise NotImplementedError('The subclass should implement this method.')

    def _add_block(self, block):
        raise NotImplementedError('The subclass should implement this method.')

    def _compile_objective(self):
        raise NotImplementedError('The subclass should implement this method.')

    def _add_constraint(self, con):
        raise NotImplementedError('The subclass should implement this method.')

    def _add_var(self, var):
        raise NotImplementedError('The subclass should implement this method.')

    def _add_sos_constraint(self, con):
        raise NotImplementedError('The subclass should implement this method.')

    def compile_instance(self, model, **kwds):
        return self._compile_instance(model)

    def add_block(self, block):
        return self._add_block(block)

    def compile_objective(self):
        return self._compile_objective()

    def add_constraint(self, con):
        return self._add_constraint(con)

    def add_var(self, var):
        return self._add_var(var)

    def add_sos_constraint(self, con):
        return self._add_sos_constraint(con)

    def remove_block(self, block):
        raise NotImplementedError('The subclass should implement this method.')

    def remove_constraint(self, con):
        raise NotImplementedError('The subclass should implement this method.')

    def remove_var(self):
        raise NotImplementedError('The subclass should implement this method.')

    def available(self, exception_flag=True):
        raise NotImplementedError('The subclass should implement this method.')

    def _get_version(self):
        raise NotImplementedError('The subclass should implement this method.')