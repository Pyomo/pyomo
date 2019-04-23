from __future__ import division

import logging
import sys
import time
import timeit
import traceback
from pathos.pools import _ProcessPool as Pool

from pyomo.contrib.gdpopt.data_class import GDPoptSolveData
from pyomo.contrib.gdpopt.util import create_utility_block, copy_var_list_values
from pyomo.core import Suffix
from pyomo.common.config import (
    ConfigBlock, ConfigValue, add_docstring_list
)
from pyomo.opt import SolverFactory, SolverResults
from pyomo.opt import TerminationCondition as tc

logger = logging.getLogger('pyomo.contrib.multistart')


@SolverFactory.register('multisolve',
                        doc='Naive Parallel Solver Harness for Pyomo')
class Multisolve(object):
    """Solver wrapper that spawns multiple parallel attempts at solution.

    CAUTION: this is not very stable code because parallel processing gets tricky.

    Keyword arguments below are specified for the ``solve`` function.

    """

    CONFIG = ConfigBlock()
    CONFIG.declare("solvers", ConfigValue(
        default=['ipopt', ], domain=list,
        description="Specify the list of solvers to use.",
    ))
    CONFIG.declare("solver_args", ConfigValue(
        default=[{}, ],
        description="List of dictionary of keyword arguments to pass to the solver."
    ))
    CONFIG.declare("time_limit", ConfigValue(
        default=float('inf'),
        description="Time limit on execution (seconds)"
    ))
    CONFIG.declare("integer_tolerance", ConfigValue(
        default=1e-4,
        description="Tolerance on integrality"
    ))

    __doc__ = add_docstring_list(__doc__, CONFIG)

    def available(self, exception_flag=True):
        """Check if solver is available.

        TODO: For now, it is always available. However, sub-solvers may not
        always be available, and so this should reflect that possibility.

        """
        return True

    def solve(self, model, **kwds):
        config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)

        solve_data = GDPoptSolveData()
        with create_utility_block(model, 'multisolve_utils', solve_data):
            results = []

            with Pool(processes=2, maxtasksperchild=1) as pool:
                for solver, solver_args in zip(config.solvers, config.solver_args):
                    results.append(pool.apply_async(
                        solve_model, args=(model.clone(), solver, solver_args)))
                pool.close()
                start_time = timeit.default_timer()
                solution_found = False
                final_result = None

                # Wait for a feasible result
                while results:
                    time.sleep(0.1)
                    finished_results = [r for r in results if r.ready()]
                    successsful_results = [r for r in finished_results if r.successful()]
                    for result in successsful_results:
                        sol_result, solved_model = result.get()
                        t_cond = sol_result.solver.termination_condition
                        if any(t_cond == cond for cond in (tc.optimal, tc.feasible, tc.locallyOptimal)):
                            final_result = sol_result
                            # copy over the variable values
                            copy_var_list_values(
                                solved_model.multisolve_utils.variable_list, model.multisolve_utils.variable_list,
                                config
                            )
                            # Copy over the duals as well
                            if hasattr(solved_model, 'dual') and isinstance(solved_model.dual, Suffix):
                                for constr, orig_constr in zip(solved_model.multisolve_utils.constraint_list,
                                                               model.multisolve_utils.constraint_list):
                                    model.dual[orig_constr] = solved_model.dual.get(constr, None)
                            solution_found = True
                            break
                        elif t_cond == tc.maxTimeLimit:
                            pass  # TODO handle feasible solution but max time limit
                        elif any(t_cond == cond for cond in (tc.infeasible,)):
                            if final_result is None:
                                final_result = sol_result
                                copy_var_list_values(
                                    solved_model.multisolve_utils.variable_list, model.multisolve_utils.variable_list,
                                    config
                                )
                        else:
                            pass
                            # raise NotImplementedError("Unhandled termination condition: %s" % t_cond)
                    if solution_found:
                        break
                    elapsed = timeit.default_timer() - start_time
                    if elapsed >= config.time_limit:
                        break
                    results = [r for r in results if r not in finished_results]
                del results
                pool.terminate()
                time.sleep(0.1)
                pool.join()

                if not solution_found and final_result is not None:
                    pass
                if final_result is None:
                    final_result = SolverResults()
                    final_result.solver.termination_condition = tc.maxTimeLimit
                return final_result

    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass


def solve_model(m, solver, solver_args):
    result = SolverFactory(solver).solve(m, **solver_args)
    return result, m
