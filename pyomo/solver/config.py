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

from pyomo.common.config import (
    ConfigDict,
    ConfigValue,
    NonNegativeFloat,
    NonNegativeInt,
)


class SolverConfig(ConfigDict):
    """
    Attributes
    ----------
    time_limit: float - sent to solver
        Time limit for the solver
    tee: bool
        If True, then the solver log goes to stdout
    load_solution: bool - wrapper
        If False, then the values of the primal variables will not be
        loaded into the model
    symbolic_solver_labels: bool - sent to solver
        If True, the names given to the solver will reflect the names
        of the pyomo components. Cannot be changed after set_instance
        is called.
    report_timing: bool - wrapper
        If True, then some timing information will be printed at the
        end of the solve.
    threads: integer - sent to solver
        Number of threads to be used by a solver.
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

        # TODO: Add in type-hinting everywhere
        self.tee: bool = self.declare('tee', ConfigValue(domain=bool, default=False))
        self.load_solution: bool = self.declare(
            'load_solution', ConfigValue(domain=bool, default=True)
        )
        self.symbolic_solver_labels: bool = self.declare(
            'symbolic_solver_labels', ConfigValue(domain=bool, default=False)
        )
        self.report_timing: bool = self.declare(
            'report_timing', ConfigValue(domain=bool, default=False)
        )
        self.threads = self.declare('threads', ConfigValue(domain=NonNegativeInt))
        self.time_limit: NonNegativeFloat = self.declare(
            'time_limit', ConfigValue(domain=NonNegativeFloat)
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

        self.rel_gap: NonNegativeFloat = self.declare(
            'rel_gap', ConfigValue(domain=NonNegativeFloat)
        )
        self.abs_gap: NonNegativeFloat = self.declare(
            'abs_gap', ConfigValue(domain=NonNegativeFloat)
        )
        self.relax_integrality: bool = self.declare(
            'relax_integrality', ConfigValue(domain=bool, default=False)
        )


class UpdateConfig(ConfigDict):
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

        self.declare(
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
        self.declare(
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
        self.declare(
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
        self.declare(
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
        self.declare(
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
        self.declare(
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
        self.declare(
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
        self.declare(
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
        self.declare(
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
        self.declare(
            'treat_fixed_vars_as_params',
            ConfigValue(
                domain=bool,
                default=True,
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

        self.check_for_new_or_removed_constraints: bool = True
        self.check_for_new_or_removed_vars: bool = True
        self.check_for_new_or_removed_params: bool = True
        self.check_for_new_objective: bool = True
        self.update_constraints: bool = True
        self.update_vars: bool = True
        self.update_params: bool = True
        self.update_named_expressions: bool = True
        self.update_objective: bool = True
        self.treat_fixed_vars_as_params: bool = True
