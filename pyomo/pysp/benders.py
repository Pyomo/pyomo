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
from pyomo.pysp.solvers.benders import (SolverFactory, TerminationCondition,
                                        undefined, value, minimize, Set,
                                        Objective, SOSConstraint, Constraint,
                                        Var, RangeSet, Expression, Suffix,
                                        Reals, Param, _GeneralConstraintData,
                                        XConstraintList, PySPConfiguredObject,
                                        PySPConfigValue, PySPConfigBlock,
                                        safe_register_common_option,
                                        safe_declare_common_option,
                                        safe_declare_unique_option,
                                        _domain_percent, _domain_nonnegative,
                                        _domain_positive_integer,
                                        _domain_must_be_str,
                                        _domain_unit_interval, 
                                        _domain_tuple_of_str,
                                        _domain_tuple_of_str_or_dict,
                                        InvocationType,
                                        ScenarioTreeManager,
                                        ScenarioTreeManagerFactory,
                                        ScenarioTreeManagerSolverFactory,
                                        find_active_objective, create_ef_instance,
                                        SPSolver, SPSolverResults,
                                        SPSolverFactory,
                                        EXTERNAL_deactivate_rootnode_costs,
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
