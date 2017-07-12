from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import DirectOrPersistentSolver


class DirectSolver(DirectOrPersistentSolver):
    """
    Subclasses need to:
    1.) Initialize self._solver_model during _presolve before calling DirectSolver._presolve
    """
    def __init__(self, **kwds):
        super(DirectSolver, self).__init__(**kwds)

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

        super(DirectSolver, self)._presolve(*args, **kwds)

    def _apply_solver(self):
        raise NotImplementedError('The specific direct/persistent solver interface should implement this method.')

    def _postsolve(self):
        return super(DirectSolver, self)._postsolve()

    def _compile_instance(self, model, **kwds):
        super(DirectSolver, self)._compile_instance(model, **kwds)

    def _add_block(self, block):
        raise NotImplementedError('The specific direct/persistent solver interface should implement this method.')

    def _compile_objective(self):
        raise NotImplementedError('The specific direct/persistent solver interface should implement this method.')

    def _add_constraint(self, con):
        raise NotImplementedError('The specific direct/persistent solver interface should implement this method.')

    def _add_var(self, var):
        raise NotImplementedError('The specific direct/persistent solver interface should implement this method.')

    def _get_expr_from_pyomo_repn(self, repn):
        raise NotImplementedError('The subclass should implement this method.')

    def _get_expr_from_pyomo_expr(self, expr):
        raise NotImplementedError('The specific direct/persistent solver interface should implement this method.')

    def _load_vars(self, vars_to_load):
        raise NotImplementedError('The specific direct solver interface should implement this method.')

    def warm_start_capable(self):
        return False

    def _warm_start(self):
        raise NotImplementedError('If a subclass can warmstart, then it should implement this method.')
