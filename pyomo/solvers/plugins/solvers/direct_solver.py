from pyomo.core.base.PyomoModel import Model
from pyomo.core.kernel.component_block import IBlockStorage
from pyomo.opt.base.solvers import OptSolver
from pyomo.core.base import SymbolMap, NumericLabeler, TextLabeler


class DirectSolver(OptSolver):
    """
    Subclasses need to:
    1.) Initialize self._solver_model during _presolve before calling DirectSolver._presolve
    """
    def __init__(self, **kwds):
        super(DirectSolver, self).__init__(**kwds)

        self._pyomo_model = None
        self._solver_model = None
        self._symbol_map = None
        self._labeler = None
        self._pyomo_var_to_solver_var_map = None
        self._pyomo_con_to_solver_con_map = None
        self._objective_label = None
        self.results = None
        self._smap_id = None

        # this interface doesn't use files, but we can create a log file if requested
        self._keepfiles = False

    def _presolve(self, *args, **kwds):
        model = args[0]
        self._pyomo_model = model
        if not isinstance(model, (Model, IBlockStorage)):
            msg = "The problem instance supplied to the CPLEXDirect plugin " \
                  "method '_presolve' must be of type 'Model' - "\
                  "interface does not currently support file names"
            raise ValueError(msg)

        self._symbol_map = SymbolMap()
        self._smap_id = id(self._symbol_map)
        self._pyomo_var_to_solver_var_map = {}
        self._pyomo_con_to_solver_con_map = {}
        symbolic_solver_labels = kwds.pop('symbolic_solver_labels', False)
        if symbolic_solver_labels:
            self._labeler = TextLabeler()
        else:
            self._labeler = NumericLabeler('x')

        self._compile_instance(model)

        super(DirectSolver, self)._presolve(*args, **kwds)

    def _apply_solver(self):
        raise NotImplementedError('The specific direct/persistent solver interface should implement this method.')

    def _postsolve(self):
        return super(DirectSolver, self)._postsolve()

    def _compile_instance(self, model):
        raise NotImplementedError('The specific direct/persistent solver interface should implement this method.')

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
