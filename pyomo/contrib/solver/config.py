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

from typing import Optional

from pyomo.common.config import (
    ConfigDict,
    ConfigValue,
    NonNegativeFloat,
    NonNegativeInt,
    ADVANCED_OPTION,
)
from pyomo.common.timing import HierarchicalTimer


class AutoUpdateConfig(ConfigDict):
    """
    This is necessary for persistent solvers.

    Attributes
    ----------
    check_for_new_or_removed_constraints: bool
    check_for_new_or_removed_vars: bool
    check_for_new_or_removed_params: bool
    update_constraints: bool
    update_vars: bool
    update_params: bool
    update_named_expressions: bool
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
                doc="""
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
                doc="""
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
                doc="""
                If False, new/old parameters will not be automatically detected on subsequent 
                solves. Use False only when manually updating the solver with opt.add_params() and 
                opt.remove_params() or when you are certain parameters are not being added to /
                removed from the model.""",
            ),
        )
        self.check_for_new_objective: bool = self.declare(
            'check_for_new_objective',
            ConfigValue(
                domain=bool,
                default=True,
                doc="""
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
                doc="""
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
                doc="""
                If False, changes to existing variables will not be automatically detected on 
                subsequent solves. This includes changes to the lb, ub, domain, and fixed 
                attributes of variables. Use False only when manually updating the solver with 
                opt.update_variables() or when you are certain variables are not being modified.""",
            ),
        )
        self.update_params: bool = self.declare(
            'update_params',
            ConfigValue(
                domain=bool,
                default=True,
                doc="""
                If False, changes to parameter values will not be automatically detected on 
                subsequent solves. Use False only when manually updating the solver with 
                opt.update_params() or when you are certain parameters are not being modified.""",
            ),
        )
        self.update_named_expressions: bool = self.declare(
            'update_named_expressions',
            ConfigValue(
                domain=bool,
                default=True,
                doc="""
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
                doc="""
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
                doc="""
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


class SolverConfig(ConfigDict):
    """
    Base config values for all solver interfaces
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

        self.tee: bool = self.declare(
            'tee',
            ConfigValue(
                domain=bool,
                default=False,
                description="If True, the solver log prints to stdout.",
            ),
        )
        self.load_solution: bool = self.declare(
            'load_solution',
            ConfigValue(
                domain=bool,
                default=True,
                description="If True, the values of the primal variables will be loaded into the model.",
            ),
        )
        self.raise_exception_on_nonoptimal_result: bool = self.declare(
            'raise_exception_on_nonoptimal_result',
            ConfigValue(
                domain=bool,
                default=True,
                description="If False, the `solve` method will continue processing even if the returned result is nonoptimal.",
            ),
        )
        self.symbolic_solver_labels: bool = self.declare(
            'symbolic_solver_labels',
            ConfigValue(
                domain=bool,
                default=False,
                description="If True, the names given to the solver will reflect the names of the Pyomo components. Cannot be changed after set_instance is called.",
            ),
        )
        self.timer: HierarchicalTimer = self.declare(
            'timer',
            ConfigValue(
                default=None,
                description="A HierarchicalTimer.",
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
                domain=NonNegativeFloat, description="Time limit applied to the solver."
            ),
        )
        self.solver_options: ConfigDict = self.declare(
            'solver_options',
            ConfigDict(implicit=True, description="Options to pass to the solver."),
        )


class BranchAndBoundConfig(SolverConfig):
    """
    Attributes
    ----------
    mip_gap: float
        Solver will terminate if the mip gap is less than mip_gap
    relax_integrality: bool
        If True, all integer variables will be relaxed to continuous
        variables before solving
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
            'rel_gap', ConfigValue(domain=NonNegativeFloat)
        )
        self.abs_gap: Optional[float] = self.declare(
            'abs_gap', ConfigValue(domain=NonNegativeFloat)
        )


class PersistentSolverConfig(SolverConfig):
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

        self.auto_updats: AutoUpdateConfig = self.declare('auto_updates', AutoUpdateConfig())


class PersistentBranchAndBoundConfig(BranchAndBoundConfig):
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

        self.auto_updats: AutoUpdateConfig = self.declare('auto_updates', AutoUpdateConfig())
