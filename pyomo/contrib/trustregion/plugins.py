
from pyutilib.services import TempfileManager
from pyutilib.services import register_executable

from pyomo.opt.base import SolverFactory, OptSolver
from pyomo.solvers.plugins.solvers.ASL import ASL

from pyomo.contrib.trustregion.TRF import TRF
from pyomo.contrib.trustregion.readgjh import readgjh
import pyomo.contrib.trustregion.param as param

def load():
    pass

@SolverFactory.register(
        'trustregion',
        doc='Trust region filter method for black box/glass box optimization'
    )
class TrustRegionSolver(OptSolver):
    """
    A trust region filter method for black box / glass box optimizaiton
    Solves nonlinear optimization problems containing external function calls
    through automatic construction of reduced models (ROM), also known as
    surrogate models.
    Currently implements linear and quadratic reduced models.
    See Eason, Biegler (2016) AIChE Journal for more details

    Arguments:
    """
    #    + param.CONFIG.generte_yaml_template()

    def __init__(self, **kwds):
        #
        # Call base class constructor
        #
        kwds['type'] = 'trustregion'
        OptSolver.__init__(self, **kwds)

    def available(self, exception_flag=True):
        """Check if solver is available.

        TODO: For now, it is always available. However, sub-solvers may not
        always be available, and so this should reflect that possibility.

        """
        return True

    def version(self):
        """Return a 3-tuple describing the solver version."""
        return __version__

    def solve(self, model, eflist, **kwds):
        assert not kwds
        #config = param.CONFIG(kwds)
        return TRF(model, eflist)#, config)


@SolverFactory.register('contrib.gjh', doc='Interface to the AMPL GJH "solver"')
class GJHSolver(ASL):
    """An interface to the AMPL GJH "solver" for evaluating a model at a
    point."""

    def __init__(self, **kwds):
        kwds['type'] = 'gjh'
        kwds['symbolic_solver_labels'] = True
        super(GJHSolver, self).__init__(**kwds)
        self.options.solver = 'gjh'
        self._metasolver = False

    # A hackish way to hold on to the model so that we can parse the
    # results.
    def _initialize_callbacks(self, model):
        self._model = model
        self._model._gjh_info = None
        super(GJHSolver, self)._initialize_callbacks(model)

    def _presolve(self, *args, **kwds):
        super(GJHSolver, self)._presolve(*args, **kwds)
        self._gjh_file = self._soln_file[:-3]+'gjh'
        TempfileManager.add_tempfile(self._gjh_file, exists=False)

    def _postsolve(self):
        #
        # TODO: We should return the information using a better data
        # structure (ComponentMap? so that the GJH solver does not need
        # to be called with symbolic_solver_labels=True
        #
        self._model._gjh_info = readgjh(self._gjh_file)
        self._model = None
        return super(GJHSolver, self)._postsolve()
