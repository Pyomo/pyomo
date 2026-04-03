#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022 Pyomo Developers
#  Copyright (c) 2017-2022 National Technology and Engineering Solutions of
#  Sandia, LLC. Under the terms of Contract DE-NA0003525 with National
#  Technology and Engineering Solutions of Sandia, LLC, the U.S. Government
#  retains certain rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
import os
import sys
from pyomo.common.config import ConfigBlock, ConfigValue, In, Path
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base import SymbolMap
from pyomo.core.kernel.objective import minimize
from pyomo.opt.base import OptSolver
from pyomo.opt.results import ResultsFormat, SolutionStatus, SolverStatus
from pyomo.opt.solver import SolverFactory
from pyomo.solvers.plugins.solvers.direct_solver import DirectSolver
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver

logger = logging.getLogger('pyomo.contrib.pynumero')


@SolverFactory.register(
    'cyipopt',
    doc='The Pynumero-based interface to the Ipopt optimization solver.',
)
class IPOPT(OptSolver):
    """
    A Pynumero-based interface to Ipopt.
    """

    def __init__(self, **kwds):
        super(IPOPT, self).__init__(**kwds)
        self._valid_problem_formats = []
        self._valid_result_formats = {ResultsFormat.sol}
        self._capabilities.linear = False
        self._smap_id = None
        self._pyomo_model = None
        self._results = None
        self._tee = False
        self._solver_options = {}

        # Set up the solver config
        self.config.declare(
            'solver_io',
            ConfigValue(
                default='nl',
                domain=In(['nl']),
                description="The solver interface to use.",
            ),
        )
        self.config.declare(
            'options',
            ConfigBlock(
                implicit=True,
                description="A dict of options to pass to the solver",
            ),
        )
        self.config.declare(
            'hessian_approximation',
            ConfigValue(
                default='exact',
                domain=In(['exact', 'limited-memory']),
                description=(
                    "The method to use for approximating the Hessian. "
                    "'exact' requires the ASL interface and a compiled "
                    "suffixed library. 'limited-memory' uses the python "
                    "nlp interface."
                ),
            ),
        )

    def available(self, exception_flag=True):
        """
        Check if the solver is available.

        In this case, it checks for the presense of the cyipopt
        and numpy python packages.
        """
        try:
            import numpy
            from pyomo.contrib.pynumero.interfaces.cyipopt_interface import (
                cyipopt_available,
            )
        except ImportError:
            return False
        return cyipopt_available

    def solve(self, model, **kwds):
        """
        Solve the model.

        Arguments
        ---------
        model: ConcreteModel
            The Pyomo model to be solved
        tee: bool
            If true, stream the solver output
        """
        from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
        from pyomo.contrib.pynumero.interfaces.nlp_projections import (
            NLPWithGreyBoxBlocks,
        )
        from pyomo.contrib.pynumero.interfaces.external_grey_box import (
            ExternalGreyBoxBlock,
        )

        self.config.set_value(kwds)
        self._tee = self.config.tee

        # Check for grey-box models that do not provide Hessians
        # and automatically switch to limited-memory approximation
        # if the user has not specified an approximation method.
        if not self.config.get('hessian_approximation').is_set_by_user():
            provides_full_hessian = True
            has_grey_box = False
            for blk in model.component_data_objects(
                ExternalGreyBoxBlock, active=True, descend_into=True
            ):
                has_grey_box = True
                ex_model = blk.get_external_model()
                if not ex_model:
                    continue

                if ex_model.n_equality_constraints() > 0 and not hasattr(
                    ex_model, 'evaluate_hessian_equality_constraints'
                ):
                    provides_full_hessian = False
                    break
                if ex_model.n_outputs() > 0 and not hasattr(
                    ex_model, 'evaluate_hessian_outputs'
                ):
                    provides_full_hessian = False
                    break
                if ex_model.has_objective() and not hasattr(
                    ex_model, 'evaluate_hessian_objective'
                ):
                    provides_full_hessian = False
                    break

            if has_grey_box and not provides_full_hessian:
                logger.info(
                    "A grey-box model without full Hessian support has been "
                    "detected. Setting hessian_approximation to "
                    "'limited-memory'."
                )
                self.config.hessian_approximation = 'limited-memory'

        if self.config.hessian_approximation == 'exact':
            if not NLPWithGreyBoxBlocks.available():
                raise RuntimeError(
                    "The cyipopt solver requires nlp_solvers to be available."
                )
            nlp = PyomoNLP(model)
            if nlp.get_external_grey_box_models():
                nlp = NLPWithGreyBoxBlocks(nlp)
        else:
            # The ASL python interface does not support grey box models
            # so we must use the PyomoNLP
            if not NLPWithGreyBoxBlocks.available():
                raise RuntimeError(
                    "The cyipopt solver requires nlp_solvers to be available."
                )
            nlp = PyomoNLP(model)
            if nlp.get_external_grey_box_models():
                nlp = NLPWithGreyBoxBlocks(nlp)

        from pyomo.contrib.pynumero.interfaces.cyipopt_interface import (
            PyomoCyIpoptProblem,
        )

        # create the ipopt problem
        ipopt_problem = PyomoCyIpoptProblem(
            nlp=nlp, tee=self._tee, options=self.config.options
        )

        # solve the problem
        x, info = ipopt_problem.solve()

        # load the results
        self._results = ipopt_problem.results
        self._pyomo_model = model
        self._smap_id = ipopt_problem.symbol_map.id
        self.load_vars()
        self.load_duals()

        return self._results

    def _postsolve(self):
        self._pyomo_model = None
        self._smap_id = None
        return self._results
