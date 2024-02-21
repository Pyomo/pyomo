#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import io
import logging
import sys

from collections.abc import Sequence
from typing import Optional, List, TextIO

from pyomo.common.config import (
    ConfigDict,
    ConfigValue,
    NonNegativeFloat,
    NonNegativeInt,
    ADVANCED_OPTION,
    Bool,
    Path,
)
from pyomo.common.log import LogStream
from pyomo.common.numeric_types import native_logical_types
from pyomo.common.timing import HierarchicalTimer


def TextIO_or_Logger(val):
    ans = []
    if not isinstance(val, Sequence):
        val = [val]
    for v in val:
        if v.__class__ in native_logical_types:
            if v:
                ans.append(sys.stdout)
        elif isinstance(v, io.TextIOBase):
            ans.append(v)
        elif isinstance(v, logging.Logger):
            ans.append(LogStream(level=logging.INFO, logger=v))
        else:
            raise ValueError(
                "Expected bool, TextIOBase, or Logger, but received {v.__class__}"
            )
    return ans


class SolverConfig(ConfigDict):
    """
    Base config for all direct solver interfaces
    """

    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super().__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )

        self.tee: List[TextIO] = self.declare(
            'tee',
            ConfigValue(
                domain=TextIO_or_Logger,
                default=False,
                description="""``tee`` accepts :py:class:`bool`,
                :py:class:`io.TextIOBase`, or :py:class:`logging.Logger`
                (or a list of these types).  ``True`` is mapped to
                ``sys.stdout``.  The solver log will be printed to each of
                these streams / destinations.""",
            ),
        )
        self.working_dir: Optional[Path] = self.declare(
            'working_dir',
            ConfigValue(
                domain=Path(),
                default=None,
                description="The directory in which generated files should be saved. "
                "This replaces the `keepfiles` option.",
            ),
        )
        self.load_solutions: bool = self.declare(
            'load_solutions',
            ConfigValue(
                domain=Bool,
                default=True,
                description="If True, the values of the primal variables will be loaded into the model.",
            ),
        )
        self.raise_exception_on_nonoptimal_result: bool = self.declare(
            'raise_exception_on_nonoptimal_result',
            ConfigValue(
                domain=Bool,
                default=True,
                description="If False, the `solve` method will continue processing "
                "even if the returned result is nonoptimal.",
            ),
        )
        self.symbolic_solver_labels: bool = self.declare(
            'symbolic_solver_labels',
            ConfigValue(
                domain=Bool,
                default=False,
                description="If True, the names given to the solver will reflect the names of the Pyomo components. "
                "Cannot be changed after set_instance is called.",
            ),
        )
        self.timer: Optional[HierarchicalTimer] = self.declare(
            'timer',
            ConfigValue(
                default=None,
                description="A timer object for recording relevant process timing data.",
            ),
        )
        self.threads: Optional[int] = self.declare(
            'threads',
            ConfigValue(
                domain=NonNegativeInt,
                description="Number of threads to be used by a solver.",
                default=None,
            ),
        )
        self.time_limit: Optional[float] = self.declare(
            'time_limit',
            ConfigValue(
                domain=NonNegativeFloat,
                description="Time limit applied to the solver (in seconds).",
            ),
        )
        self.solver_options: ConfigDict = self.declare(
            'solver_options',
            ConfigDict(implicit=True, description="Options to pass to the solver."),
        )


class BranchAndBoundConfig(SolverConfig):
    """
    Base config for all direct MIP solver interfaces

    Attributes
    ----------
    rel_gap: float
        The relative value of the gap in relation to the best bound
    abs_gap: float
        The absolute value of the difference between the incumbent and best bound
    """

    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super().__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )

        self.rel_gap: Optional[float] = self.declare(
            'rel_gap',
            ConfigValue(
                domain=NonNegativeFloat,
                description="Optional termination condition; the relative value of the "
                "gap in relation to the best bound",
            ),
        )
        self.abs_gap: Optional[float] = self.declare(
            'abs_gap',
            ConfigValue(
                domain=NonNegativeFloat,
                description="Optional termination condition; the absolute value of the "
                "difference between the incumbent and best bound",
            ),
        )


class AutoUpdateConfig(ConfigDict):
    """
    This is necessary for persistent solvers.

    Attributes
    ----------
    check_for_new_or_removed_constraints: bool
    check_for_new_or_removed_vars: bool
    check_for_new_or_removed_params: bool
    check_for_new_objective: bool
    update_constraints: bool
    update_vars: bool
    update_parameters: bool
    update_named_expressions: bool
    update_objective: bool
    treat_fixed_vars_as_params: bool
    """

    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        if doc is None:
            doc = 'Configuration options to detect changes in model between solves'
        super().__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )

        self.check_for_new_or_removed_constraints: bool = self.declare(
            'check_for_new_or_removed_constraints',
            ConfigValue(
                domain=bool,
                default=True,
                description="""
                If False, new/old constraints will not be automatically detected on subsequent
                solves. Use False only when manually updating the solver with opt.add_constraints()
                and opt.remove_constraints() or when you are certain constraints are not being
                added to/removed from the model.""",
            ),
        )
        self.check_for_new_or_removed_vars: bool = self.declare(
            'check_for_new_or_removed_vars',
            ConfigValue(
                domain=bool,
                default=True,
                description="""
                If False, new/old variables will not be automatically detected on subsequent 
                solves. Use False only when manually updating the solver with opt.add_variables() and 
                opt.remove_variables() or when you are certain variables are not being added to /
                removed from the model.""",
            ),
        )
        self.check_for_new_or_removed_params: bool = self.declare(
            'check_for_new_or_removed_params',
            ConfigValue(
                domain=bool,
                default=True,
                description="""
                If False, new/old parameters will not be automatically detected on subsequent 
                solves. Use False only when manually updating the solver with opt.add_parameters() and 
                opt.remove_parameters() or when you are certain parameters are not being added to /
                removed from the model.""",
            ),
        )
        self.check_for_new_objective: bool = self.declare(
            'check_for_new_objective',
            ConfigValue(
                domain=bool,
                default=True,
                description="""
                If False, new/old objectives will not be automatically detected on subsequent 
                solves. Use False only when manually updating the solver with opt.set_objective() or 
                when you are certain objectives are not being added to / removed from the model.""",
            ),
        )
        self.update_constraints: bool = self.declare(
            'update_constraints',
            ConfigValue(
                domain=bool,
                default=True,
                description="""
                If False, changes to existing constraints will not be automatically detected on 
                subsequent solves. This includes changes to the lower, body, and upper attributes of 
                constraints. Use False only when manually updating the solver with 
                opt.remove_constraints() and opt.add_constraints() or when you are certain constraints 
                are not being modified.""",
            ),
        )
        self.update_vars: bool = self.declare(
            'update_vars',
            ConfigValue(
                domain=bool,
                default=True,
                description="""
                If False, changes to existing variables will not be automatically detected on 
                subsequent solves. This includes changes to the lb, ub, domain, and fixed 
                attributes of variables. Use False only when manually updating the solver with 
                opt.update_variables() or when you are certain variables are not being modified.""",
            ),
        )
        self.update_parameters: bool = self.declare(
            'update_parameters',
            ConfigValue(
                domain=bool,
                default=True,
                description="""
                If False, changes to parameter values will not be automatically detected on 
                subsequent solves. Use False only when manually updating the solver with 
                opt.update_parameters() or when you are certain parameters are not being modified.""",
            ),
        )
        self.update_named_expressions: bool = self.declare(
            'update_named_expressions',
            ConfigValue(
                domain=bool,
                default=True,
                description="""
                If False, changes to Expressions will not be automatically detected on 
                subsequent solves. Use False only when manually updating the solver with 
                opt.remove_constraints() and opt.add_constraints() or when you are certain 
                Expressions are not being modified.""",
            ),
        )
        self.update_objective: bool = self.declare(
            'update_objective',
            ConfigValue(
                domain=bool,
                default=True,
                description="""
                If False, changes to objectives will not be automatically detected on 
                subsequent solves. This includes the expr and sense attributes of objectives. Use 
                False only when manually updating the solver with opt.set_objective() or when you are 
                certain objectives are not being modified.""",
            ),
        )
        self.treat_fixed_vars_as_params: bool = self.declare(
            'treat_fixed_vars_as_params',
            ConfigValue(
                domain=bool,
                default=True,
                visibility=ADVANCED_OPTION,
                description="""
                This is an advanced option that should only be used in special circumstances. 
                With the default setting of True, fixed variables will be treated like parameters. 
                This means that z == x*y will be linear if x or y is fixed and the constraint 
                can be written to an LP file. If the value of the fixed variable gets changed, we have 
                to completely reprocess all constraints using that variable. If 
                treat_fixed_vars_as_params is False, then constraints will be processed as if fixed 
                variables are not fixed, and the solver will be told the variable is fixed. This means 
                z == x*y could not be written to an LP file even if x and/or y is fixed. However, 
                updating the values of fixed variables is much faster this way.""",
            ),
        )


class PersistentSolverConfig(SolverConfig):
    """
    Base config for all persistent solver interfaces
    """

    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super().__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )

        self.auto_updates: AutoUpdateConfig = self.declare(
            'auto_updates', AutoUpdateConfig()
        )


class PersistentBranchAndBoundConfig(BranchAndBoundConfig):
    """
    Base config for all persistent MIP solver interfaces
    """

    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super().__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )

        self.auto_updates: AutoUpdateConfig = self.declare(
            'auto_updates', AutoUpdateConfig()
        )
