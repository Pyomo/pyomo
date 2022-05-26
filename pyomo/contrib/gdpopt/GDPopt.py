#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Main driver module for GDPopt solver.

22.5.13 changes:
- rewrite of all algorithms
- deprecate 'strategy' in favor of 'algorithm'
- deprecate 'init_strategy' in favor of 'init_algorithm'
20.2.28 changes:
- bugfixes on tests
20.1.22 changes:
- improved subsolver time limit support for GAMS interface
- add maxTimeLimit exit condition for GDPopt-LBB
- add token Big M for reactivated constraints in GDPopt-LBB
- activate fbbt for branch-and-bound nodes
20.1.15 changes:
- internal cleanup of codebase
- merge GDPbb capabilities (logic-based branch and bound)
- refactoring of GDPbb code
- update logging information to include subsolver options
- improve SuppressInfeasibleWarning
- simplify mip preprocessing
- remove not-fully-implemented 'backtracking' from LOA
19.10.11 changes:
- bugfix on SolverStatus error message
19.5.13 changes:
- add handling to integer cuts for disjunct pruning during FBBT
19.4.23 changes:
- add support for linear subproblems
- use automatic differentiation for large constraints
- bugfixes on time limit support
- treat fixed variables as constants in GLOA cut generation
19.3.25 changes:
- add rudimentary time limit support
- start keeping basic changelog

"""
from io import StringIO
from textwrap import indent

from pyomo.common.config import (
    add_docstring_list, In, ConfigBlock, ConfigValue, NonNegativeInt,
    PositiveInt)
from pyomo.common.deprecation import deprecation_warning
from pyomo.contrib.gdpopt.branch_and_bound import _GDP_LBB_Solver
from pyomo.contrib.gdpopt.gloa import _GDP_GLOA_Solver
from pyomo.contrib.gdpopt.loa import _GDP_LOA_Solver
from pyomo.contrib.gdpopt.ric import _GDP_RIC_Solver
from pyomo.contrib.gdpopt.util import a_logger, lower_logger_level_to
from pyomo.opt.base import SolverFactory

__version__ = (22, 5, 13)  # Note: date-based version number
_supported_algorithms = {
    'LOA': (_GDP_LOA_Solver, 'Logic-based Outer Approximation'),
    'GLOA': (_GDP_GLOA_Solver, 'Global Logic-based Outer Approximation'),
    'LBB': (_GDP_LBB_Solver, 'Logic-based Branch and Bound'),
    'RIC': (_GDP_RIC_Solver, 'Relaxation with Integer Cuts')
}

def _strategy_deprecation(strategy):
    deprecation_warning("The argument 'strategy' has been deprecated "
                        "in favor of 'algorithm.'", version="TBD")
    return In(_supported_algorithms.keys())(strategy)

def _handle_strategy_deprecation(config):
    if config.algorithm is None and config.strategy is not None:
        config.algorithm = config.strategy

@SolverFactory.register(
    'gdpopt',
    doc='The GDPopt decomposition-based '
    'Generalized Disjunctive Programming (GDP) solver')
class GDPoptSolver():
    """Decomposition solver for Generalized Disjunctive Programming (GDP)
    problems.

    The GDPopt (Generalized Disjunctive Programming optimizer) solver applies a
    variety of decomposition-based approaches to solve Generalized Disjunctive
    Programming (GDP) problems. GDP models can include nonlinear, continuous
    variables and constraints, as well as logical conditions.

    These approaches include:

    - Logic-based outer approximation (LOA)
    - Logic-based branch-and-bound (LBB)
    - Partial surrogate cuts [pending]
    - Generalized Bender decomposition [pending]

    This solver implementation was developed by Carnegie Mellon University in
    the research group of Ignacio Grossmann.

    For nonconvex problems, LOA may not report rigorous lower/upper bounds.

    Questions: Please make a post at StackOverflow and/or contact Qi Chen
    <https://github.com/qtothec> or David Bernal <https://github.com/bernalde>.

    Several key GDPopt components were prototyped by BS and MS students:

    - Logic-based branch and bound: Sunjeev Kale
    - MC++ interface: Johnny Bates
    - LOA set-covering initialization: Eloy Fernandez
    - Logic-to-linear transformation: Romeo Valentin

    """
    _supported_algorithms = _supported_algorithms

    _CONFIG = ConfigBlock("GDPopt")
    _CONFIG.declare("iterlim", ConfigValue(
        default=100, domain=NonNegativeInt,
        description="Iteration limit."
    ))
    _CONFIG.declare("time_limit", ConfigValue(
        default=600,
        domain=PositiveInt,
        description="Time limit (seconds, default=600)",
        doc="""
        Seconds allowed until terminated. Note that the time limit can
        currently only be enforced between subsolver invocations. You may
        need to set subsolver time limits as well."""
    ))
    _CONFIG.declare("strategy", ConfigValue(
        default=None, domain=_strategy_deprecation,
        description="DEPRECATED: Please use 'algorithm' instead."
    ))
    _CONFIG.declare("algorithm", ConfigValue(
        default=None, domain=In(_supported_algorithms.keys()),
        description="Algorithm to use."
    ))
    _CONFIG.declare("tee", ConfigValue(
        default=False,
        description="Stream output to terminal.",
        domain=bool
    ))
    _CONFIG.declare("logger", ConfigValue(
        default='pyomo.contrib.gdpopt',
        description="The logger object or name to use for reporting.",
        domain=a_logger
    ))
    _impl = None

    @property
    def _implementation(self):
        """This property returns the class set as _impl if it is set.
        Otherwise, we check if the instance config.algorithm has been set.
        (This could happen if someone has been setting config options directly
        rather than in the constructor.) If it has, we set _impl and return it.
        """
        if self._impl is not None:
            return self._impl # it exists, we'll just return it

        # bypass the getter because we know we don't have an _impl (and calling
        # it will actually be an infinite loop since it calls this property.)
        elif self._CONFIG.algorithm is not None:
            # Check CONFIG block for an algorithm we don't know about yet
            self._impl = _supported_algorithms[
                self._CONFIG.algorithm][0](self)
        return self._impl

    @_implementation.setter
    def _implementation(self, impl):
        self._impl = impl

    @property
    def CONFIG(self):
        # Here we play a game where, if _impl is not set, we return this
        # instance's config block. However, if the _impl is set, then we have
        # transfered everything from here onto its config block, plus it has its
        # own things. So we return that block as if its this one's.
        impl = self._implementation
        if impl is None:
            return self._CONFIG
        return impl.CONFIG

    @property
    def iteration(self):
        impl = self._implementation
        if impl is not None:
            return impl.iteration + impl.initialization_iteration
        else:
            # Or give an error? I'm not sure.
            return None

    def __init__(self, **kwds):
        self._CONFIG = self._CONFIG(kwds.pop('options', {}),
                                    preserve_implicit=True)
        # First, we ignore args we don't recognize: we'll do a second pass after
        # we've set the algorithm. (This is becasue there are many
        # algorithm-specific options, like 'mip_solver', so we don't want to
        # complain *now* that we don't know about them, if they turn out to be
        # rational. But if an algorithm that uses a 'mip_solver' isn't
        # specified, we are free to pitch a fit.)
        self._CONFIG.set_value(kwds, skip_implicit=True)
        # [ESJ 5/22/22] This can go away when the strategy arg is removed.
        _handle_strategy_deprecation(self._CONFIG)

        # Now, check for an algorithm, and set _impl if we have one. We pass the
        # kwds to the constructor for the algorithm class because now it can set
        # the options that we may have ignored above.
        if self._CONFIG.algorithm is not None:
            self._implementation = _supported_algorithms[
                self._CONFIG.algorithm][0](self, kwds)

    def solve(self, model, **kwds):
        """Solve the model.

        Args:
            model (Block): a Pyomo model or block to be solved

        """
        # The algorithm can be specified as an argument to the solve method, so
        # if that's different we might have to cache what the instance has as
        # the algorithm and restore impl after we're done.
        old_impl = None
        # If we set or change the algorithm, we are going to have to set the
        # config block based on the kwd arguments it allows. This bool tracks if
        # we need to do that.
        config_needs_set = False
        impl = self._implementation
        if impl is not None:
            # First, check if we need to change impl for this solve
            algorithm = impl.CONFIG.algorithm
            config = impl.CONFIG(kwds.pop('options', {}),
                                 preserve_implicit=True)
            config.set_value(kwds)
            _handle_strategy_deprecation(config)
            if config.algorithm != algorithm:
                # The user changed options and _impl is wrong.
                old_impl = impl
                self._implementation = _supported_algorithms[
                    config.algorithm][0](self)
                config_needs_needs_set = True
        else:
            # impl is not set: the user had better have specified the algorithm
            # in the call to solve. We parse what was passed here so that we can
            # find the algorithm
            _CONFIG = self._CONFIG(kwds.pop('options', {}),
                                   preserve_implicit=True)
            # Similarly to in the init method: There's probably extra stuff in
            # kwds, but we ignore it because we'll deal with it once we've set
            # impl and know whether or not it's valid
            _CONFIG.set_value(kwds, skip_implicit=True)
            _handle_strategy_deprecation(_CONFIG)
            # Set impl and parse the rest of the config arguments
            old_impl = impl
            self._implementation = _supported_algorithms[
                _CONFIG.algorithm][0](self)
            config_needs_set = True
        if config_needs_set:
            # impl changed, so we need to get the kwd arguments from the new
            # impl
            config = self._implementation.CONFIG(kwds.pop('options', {}),
                                                 preserve_implicit=True)
            config.set_value(kwds)
            _handle_strategy_deprecation(config)

        # For the algorithms that support init_algorithm, do the right thing if
        # the user used the deprecated arg
        if (config.algorithm in {'LOA', 'GLOA', 'RIC'} and config.init_strategy
            is not None and config.init_algorithm is None):
            config.init_algorithm = config.init_strategy

        with lower_logger_level_to(config.logger, tee=config.tee):
            self._log_solver_intro_message(config)

            # Finally, we can solve the problem
            results = self._impl._call_main_loop(model, config)

        # restore the old implementation. It might be None if an algorithm
        # wasn't specified to SolverFactory, or else it would be the one that
        # *was* specified to SolverFactory, which should be the 'default' until
        # it gets changed.
        self._implementation = old_impl

        return results

    # Support use as a context manager under current solver API
    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass

    def available(self, exception_flag=True):
        """Solver is always available. Though subsolvers may not be, they will
        raise an error when the time comes.
        """
        return True

    def license_is_valid(self):
        return True

    def version(self):
        """Return a 3-tuple describing the solver version."""
        return __version__

    def _log_solver_intro_message(self, config):
        config.logger.info(
            "Starting GDPopt version %s using %s algorithm"
            % (".".join(map(str, self.version())), config.algorithm)
        )
        mip_args_output = StringIO()
        nlp_args_output = StringIO()
        minlp_args_output = StringIO()
        lminlp_args_output = StringIO()
        config.mip_solver_args.display(ostream=mip_args_output)
        config.nlp_solver_args.display(ostream=nlp_args_output)
        config.minlp_solver_args.display(ostream=minlp_args_output)
        config.local_minlp_solver_args.display(ostream=lminlp_args_output)
        mip_args_text = indent(mip_args_output.getvalue().rstrip(), prefix=" " *
                               2 + " - ")
        nlp_args_text = indent(nlp_args_output.getvalue().rstrip(), prefix=" " *
                               2 + " - ")
        minlp_args_text = indent(minlp_args_output.getvalue().rstrip(),
                                 prefix=" " * 2 + " - ")
        lminlp_args_text = indent(lminlp_args_output.getvalue().rstrip(),
                                  prefix=" " * 2 + " - ")
        mip_args_text = "" if len(mip_args_text.strip()) == 0 else \
                        "\n" + mip_args_text
        nlp_args_text = "" if len(nlp_args_text.strip()) == 0 else \
                        "\n" + nlp_args_text
        minlp_args_text = "" if len(minlp_args_text.strip()) == 0 else \
                          "\n" + minlp_args_text
        lminlp_args_text = "" if len(lminlp_args_text.strip()) == 0 else \
                           "\n" + lminlp_args_text
        config.logger.info(
            """
            Subsolvers:
            - MILP: {milp}{milp_args}
            - NLP: {nlp}{nlp_args}
            - MINLP: {minlp}{minlp_args}
            - local MINLP: {lminlp}{lminlp_args}
            """.format(
                milp=config.mip_solver,
                milp_args=mip_args_text,
                nlp=config.nlp_solver,
                nlp_args=nlp_args_text,
                minlp=config.minlp_solver,
                minlp_args=minlp_args_text,
                lminlp=config.local_minlp_solver,
                lminlp_args=lminlp_args_text,
            ).strip()
        )
        to_cite_text = """
        If you use this software, you may cite the following:
        - Implementation:
        Chen, Q; Johnson, ES; Bernal, DE; Valentin, R; Kale, S;
        Bates, J; Siirola, JD; Grossmann, IE.
        Pyomo.GDP: an ecosystem for logic based modeling and optimization
        development.
        Optimization and Engineering, 2021.
        """.strip()
        if config.algorithm == "LOA":
            to_cite_text += "\n"
            to_cite_text += """
            - LOA algorithm:
            Türkay, M; Grossmann, IE.
            Logic-based MINLP algorithms for the optimal synthesis of process
            networks. Comp. and Chem. Eng. 1996, 20(8), 959–978.
            DOI: 10.1016/0098-1354(95)00219-7.
            """.strip()
        elif config.algorithm == "GLOA":
            to_cite_text += "\n"
            to_cite_text += """
            - GLOA algorithm:
            Lee, S; Grossmann, IE.
            A Global Optimization Algorithm for Nonconvex Generalized
            Disjunctive Programming and Applications to Process Systems.
            Comp. and Chem. Eng. 2001, 25, 1675-1697.
            DOI: 10.1016/S0098-1354(01)00732-3.
            """.strip()
        elif config.algorithm == "LBB":
            to_cite_text += "\n"
            to_cite_text += """
            - LBB algorithm:
            Lee, S; Grossmann, IE.
            New algorithms for nonlinear generalized disjunctive programming.
            Comp. and Chem. Eng. 2000, 24, 2125-2141.
            DOI: 10.1016/S0098-1354(00)00581-0.
            """.strip()
        config.logger.info(to_cite_text)

    _metasolver = False

# Modify the solve docstring to specify the specific config options that are
# possible for each algorithm. We will complain if we get ones incompatible with
# the algorithm now.
GDPoptSolver.solve.__doc__ = add_docstring_list(
    GDPoptSolver.solve.__doc__, GDPoptSolver._CONFIG, indent_by=8)
for alg_name, (alg_class, description) in _supported_algorithms.items():
    GDPoptSolver.solve.__doc__ += (" " * 8).join(
        alg_class.CONFIG.generate_documentation(
            block_start="%s Keyword Arguments\n----------------\n" % alg_name,
            block_end="",
            item_start="%s\n",
            item_body=" %s",
            item_end="",
            indent_spacing=0,
            width=256
        ).splitlines(True))
