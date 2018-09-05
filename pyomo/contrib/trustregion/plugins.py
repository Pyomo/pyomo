
from pyutilib.services import TempfileManager
from pyutilib.services import register_executable

from pyutilib.misc.config import ConfigBlock, ConfigValue
from pyomo.util.config import ( 
    PositiveInt, PositiveFloat, NonNegativeFloat, In)

from pyomo.common import plugin
from pyomo.opt.base import IOptSolver
from pyomo.solvers.plugins.solvers.ASL import ASL

from pyomo.contrib.trustregion.TRF import TRF
from pyomo.contrib.trustregion.readgjh import readgjh
import pyomo.contrib.trustregion.param as param

def load():
    pass

class TrustRegionSolver(plugin.Plugin):
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

    plugin.implements(IOptSolver)
    plugin.alias(
        'trustregion',
        doc='Trust region filter method for black box/glass box optimization'
    )

    CONFIG = ConfigBlock('Trust Region')

    # Initialize trust radius
    CONFIG.declare('trust radius', ConfigValue(
        default = 1.0,
        domain = PositiveFloat,
        description = '',
        doc = ''))

    # Initialize sample region
    CONFIG.declare('sample region', ConfigValue(
        default = True,
        domain = bool,
        description = '',
        doc = ''))

    # Initialize sample radius
    # TODO do we need to keep the if statement?
    if CONFIG.sample_region:
        default_sample_radius = 0.1
    else:
        default_sample_radius = CONFIG.trust_radius / 2.0

    CONFIG.declare('sample radius', ConfigValue(
        default = default_sample_radius,
        domain = PositiveFloat,
        description = '',
        doc = ''))

    # Initialize radius max
    CONFIG.declare('radius max', ConfigValue(
        default = 1000.0 * CONFIG.trust_radius,
        domain = PositiveFloat,
        description = '',
        doc = ''))

    # Termination tolerances
    CONFIG.declare('ep i', ConfigValue(
        default = 1e-5,
        domain = PositiveFloat,
        description = '',
        doc = ''))

    CONFIG.declare('ep delta', ConfigValue(
        default = 1e-5,
        domain = PositiveFloat,
        description = '',
        doc = ''))

    CONFIG.declare('ep chi', ConfigValue(
        default = 1e-3,
        domain = PositiveFloat,
        description = '',
        doc = ''))

    CONFIG.declare('delta min', ConfigValue(
        default = 1e-6,
        domain = PositiveFloat,
        description = 'delta min <= ep delta',
        doc = ''))

    CONFIG.declare('max it', ConfigValue(
        default = 20,
        domain = PositiveInt,
        description = '',
        doc = ''))

    # Compatibility Check Parameters
    CONFIG.declare('kappa delta', ConfigValue(
        default = 0.8,
        domain = PositiveFloat,
        description = '',
        doc = ''))

    CONFIG.declare('kappa mu', ConfigValue(
        default = 1.0,
        domain = PositiveFloat,
        description = '',
        doc = ''))

    CONFIG.declare('mu', ConfigValue(
        default = 0.5,
        domain = PositiveFloat,
        description = '',
        doc = ''))

    CONFIG.declare('ep compatibility', ConfigValue(
        default = CONFIG.ep_i,
        domain = PositiveFloat,
        description = 'Suggested value: ep compatibility == ep i',
        doc = ''))

    CONFIG.declare('compatibility penalty', ConfigValue(
        default = 0.0,
        domain = NonNegativeFloat,
        description = '',
        doc = ''))

    # Criticality Check Parameters
    CONFIG.declare('criticality check', ConfigValue(
        default = 0.1,
        domain = PositiveFloat,
        description = '',
        doc = ''))

    # Trust region update parameters
    CONFIG.declare('gamma c', ConfigValue(
        default = 0.5,
        domain = PositiveFloat,
        description = '',
        doc = ''))

    CONFIG.declare('gamma e', ConfigValue(
        default = 2.5,
        domain = PositiveFloat,
        description = '',
        doc = ''))

    # Switching Condition
    CONFIG.declare('gamma s', ConfigValue(
        default = 2.0,
        domain = PositiveFloat,
        description = '',
        doc = ''))

    CONFIG.declare('kappa theta', ConfigValue(
        default = 0.1,
        domain = PositiveFloat,
        description = '',
        doc = ''))

    CONFIG.declare('theta min', ConfigValue(
        default = 1e-4,
        domain = PositiveFloat,
        description = '',
        doc = ''))

    # Filter
    CONFIG.declare('gamma f', ConfigValue(
        default = 0.01,
        domain = PositiveFloat,
        description = 'gamma_f and gamma_theta in (0,1) are fixed parameters',
        doc = ''))

    CONFIG.declare('gamma theta', ConfigValue(
        default = 0.01,
        domain = PositiveFloat,
        description = 'gamma_f and gamma_theta in (0,1) are fixed parameters',
        doc = ''))

    CONFIG.declare('theta max', ConfigValue(
        default = 50,
        domain = PositiveInt,
        description = '',
        doc = ''))

    # Ratio test parameters (for theta steps)
    CONFIG.declare('eta1', ConfigValue(
        default = 0.05,
        domain = PositiveFloat,
        description = '',
        doc = ''))

    CONFIG.declare('eta2', ConfigValue(
        default = 0.2,
        domain = PositiveFloat,
        description = '',
        doc = ''))

    # Output level (replace with real printlevels!!!)
    CONFIG.declare('print variables', ConfigValue(
        default = False,
        domain = bool,
        description = 'TODO: remove??',
        doc = ''))

    # Sample Radius reset parameter
    CONFIG.declare('sample radius adjust', ConfigValue(
        default = 0.5,
        domain = PositiveFloat,
        description = '',
        doc = ''))

    # Default romtype
    CONFIG.declare('reduced model type', ConfigValue(
        default = 1,
        domain = In([0,1]),
        description = '0 = Linear, 1 = Quadratic',
        doc = ''))

    def __init__(self, **kwds):
        self.config = self.CONFIG(kwds)

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
        config = self.config(kwds)
        return TRF(model, eflist)

class GJHSolver(ASL):
    """An interface to the AMPL GJH "solver" for evaluating a model at a
    point."""

    plugin.implements(IOptSolver)
    plugin.alias(
        'contrib.gjh', doc='Interface to the AMPL GJH "solver"')

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
