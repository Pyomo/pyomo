from pyomo.opt.base.solvers import OptSolver


class PersistentSolver(OptSolver):

    def __init__(self, **kwds):
        """ Constructor """

        # Ensure any subclasses inherit from PersistentSolver before any direct solver
        assert type(self).__bases__[0] is PersistentSolver

        self._solver_model = None
        "The model used by the solver (i.e., not the Pyomo model)"

        # Call the __init__ method of OptSolver
        super(PersistentSolver, self).__init__(self,**kwds)

    def _presolve(self, *args, **kwds):
        super(PersistentSolver, self)._presolve(*args, **kwds)

    def compile_instance(self, model):
        return self._compile_instance(model)

    def compile_block(self, block):
        return self._compile_block(block)

    def compile_objective(self, obj):
        return self._compile_objective(obj)

    def compile_constraint(self, con):
        return self._compile_constraint(con)

    def compile_var(self, var):
        return self._compile_var(var)