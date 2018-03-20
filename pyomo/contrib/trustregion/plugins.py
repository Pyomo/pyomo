import pyomo.util.plugin
from pyomo.contrib.trustregion.TRF import TRF
from pyomo.contrib.trustregion.param import param
from pyomo.opt.base import IOptSolver



def load():
    pass

class TrustRegionSolver(pyomo.util.plugin.Plugin):
    """
    A trust region filter method for black box / glass box optimizaiton 
    Solves nonlinear optimization problems containing external function calls
    through automatic construction of reduced models (ROM), also known as 
    surrogate models. 
    Currently implements linear and quadratic reduced models.
    See Eason, Biegler (2016) AIChE Journal for more details

    Arguments:
    """ 
	#	+ param.CONFIG.generte_yaml_template()

    pyomo.util.plugin.implements(IOptSolver)
    pyomo.util.plugin.alias('trustregion',
                            doc='Trust region filter method for black box/glass box optimization')

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
        config = param.CONFIG(kwds)
        return TRF(model, eflist, config)


