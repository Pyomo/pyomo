from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import DirectOrPersistentSolver


class PersistentSolver(DirectOrPersistentSolver):

    def __init__(self, **kwds):
        super(PersistentSolver, self).__init__(**kwds)

        # Ensure any subclasses inherit from PersistentSolver before any direct solver
        assert type(self).__bases__[0] is PersistentSolver

    def _presolve(self, *args, **kwds):
        if len(args) != 0:
            msg = 'The persistent solver interface does not accept a problem instance in the solve method.'
            msg += ' The problem instance should be compiled before the solve using the compile_instance method.'
            raise ValueError(msg)

        super(PersistentSolver, self)._presolve(*args, **kwds)

    def _apply_solver(self):
        raise NotImplementedError('The subclass should implement this method.')

    def _postsolve(self):
        return super(PersistentSolver, self)._postsolve()

    def _compile_instance(self, model, **kwds):
        super(PersistentSolver, self)._compile_instance(model, **kwds)

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
        return self._compile_instance(model, **kwds)

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

    def remove_sos_constraint(self, con):
        raise NotImplementedError('The subclass should implement this method.')

    def remove_var(self, var):
        raise NotImplementedError('The subclass should implement this method.')

    def _get_expr_from_pyomo_repn(self, repn, max_degree=None):
        raise NotImplementedError('The subclass should implement this method.')

    def _get_expr_from_pyomo_expr(self, expr, max_degree=None):
        raise NotImplementedError('The subclass should implement this method.')

    def _load_vars(self, vars_to_load):
        raise NotImplementedError('The subclass should implement this method.')

    def warm_start_capable(self):
        raise NotImplementedError('The subclass should implement this method.')

    def _warm_start(self):
        raise NotImplementedError('If a subclass can warmstart, then it should implement this method.')

    def load_vars(self, vars_to_load):
        self._load_vars(vars_to_load)
