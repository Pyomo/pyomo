#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.solvers.tests.models.base

import pyomo.solvers.tests.models.LP_block
import pyomo.solvers.tests.models.LP_compiled
import pyomo.solvers.tests.models.LP_constant_objective1
import pyomo.solvers.tests.models.LP_constant_objective2
import pyomo.solvers.tests.models.LP_duals_maximize
import pyomo.solvers.tests.models.LP_duals_minimize
import pyomo.solvers.tests.models.LP_inactive_index
import pyomo.solvers.tests.models.LP_infeasible1
import pyomo.solvers.tests.models.LP_infeasible2
import pyomo.solvers.tests.models.LP_piecewise
import pyomo.solvers.tests.models.LP_simple
import pyomo.solvers.tests.models.LP_trivial_constraints
import pyomo.solvers.tests.models.LP_unbounded
import pyomo.solvers.tests.models.LP_unused_vars
# WEH - Omitting this for because it's not reliably solved by ipopt
#import pyomo.solvers.tests.models.LP_unique_duals

import pyomo.solvers.tests.models.MILP_discrete_var_bounds
import pyomo.solvers.tests.models.MILP_infeasible1
import pyomo.solvers.tests.models.MILP_simple
import pyomo.solvers.tests.models.MILP_unbounded
import pyomo.solvers.tests.models.MILP_unused_vars

import pyomo.solvers.tests.models.MIQCP_simple

import pyomo.solvers.tests.models.MIQP_simple

import pyomo.solvers.tests.models.QCP_simple

import pyomo.solvers.tests.models.QP_constant_objective
import pyomo.solvers.tests.models.QP_simple

import pyomo.solvers.tests.models.SOS1_simple
import pyomo.solvers.tests.models.SOS2_simple
