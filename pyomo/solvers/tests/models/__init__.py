#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.solvers.tests.models import (
    base,
    LP_block,
    LP_compiled,
    LP_constant_objective1,
    LP_constant_objective2,
    LP_duals_maximize,
    LP_duals_minimize,
    LP_inactive_index,
    LP_infeasible1,
    LP_infeasible2,
    LP_piecewise,
    LP_simple,
    LP_trivial_constraints,
    LP_unbounded,
    LP_unused_vars,
    # WEH - Omitting this for because it's not reliably solved by ipopt,
    # LP_unique_duals,
    MILP_discrete_var_bounds,
    MILP_infeasible1,
    MILP_simple,
    MILP_unbounded,
    MILP_unused_vars,
    MIQCP_simple,
    MIQP_simple,
    QCP_simple,
    QP_constant_objective,
    QP_simple,
    SOS1_simple,
    SOS2_simple,
)
