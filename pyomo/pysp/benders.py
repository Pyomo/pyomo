#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common import pyomo_command
from pyomo.opt import (SolverFactory,
                       TerminationCondition,
                       undefined)
from pyomo.core import (value, minimize, Set,
                        Objective, SOSConstraint,
                        Constraint, Var, RangeSet,
                        Expression, Suffix, Reals, Param)
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.beta.list_objects import XConstraintList
from pyomo.pysp.util.configured_object import PySPConfiguredObject
from pyomo.pysp.util.config import (PySPConfigValue,
                                    PySPConfigBlock,
                                    safe_register_common_option,
                                    safe_declare_common_option,
                                    safe_declare_unique_option,
                                    _domain_percent,
                                    _domain_nonnegative,
                                    _domain_positive_integer,
                                    _domain_must_be_str,
                                    _domain_unit_interval,
                                    _domain_tuple_of_str,
                                    _domain_tuple_of_str_or_dict)
from pyomo.pysp.util.misc import (parse_command_line,
                                  launch_command)
from pyomo.pysp.scenariotree.manager import \
    (InvocationType,
     ScenarioTreeManager,
     ScenarioTreeManagerFactory)
from pyomo.pysp.scenariotree.manager_solver import \
    ScenarioTreeManagerSolverFactory
from pyomo.pysp.phutils import find_active_objective
from pyomo.pysp.ef import create_ef_instance
from pyomo.pysp.solvers.spsolver import (SPSolver,
                                         SPSolverResults,
                                         SPSolverFactory)
from pyomo.pysp.solvers.benders import (EXTERNAL_deactivate_rootnode_costs,
                                        EXTERNAL_activate_rootnode_costs,
                                        EXTERNAL_activate_fix_constraints,
                                        EXTERNAL_deactivate_fix_constraints,
                                        EXTERNAL_cleanup_from_benders,
                                        EXTERNAL_initialize_for_benders,
                                        EXTERNAL_update_fix_constraints,
                                        EXTERNAL_collect_cut_data,
                                        BendersOptimalityCut, BendersAlgorithm,
                                        BendersSolver,
                                        runbenders_register_options, runbenders, 
                                        main as b_main)

@pyomo_command('runbenders', 'Optimize with the Benders solver')
def Benders_main(args=None):
    return b_main(args=args)
