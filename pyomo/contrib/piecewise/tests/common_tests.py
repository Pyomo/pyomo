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

from pyomo.core import Var
from pyomo.gdp import Disjunct, Disjunction

def check_trans_block_structure(test_case, block):
    # One (indexed) disjunct
    test_case.assertEqual(len(block.component_map(Disjunct)), 1)
    # One disjunction
    test_case.assertEqual(len(block.component_map(Disjunction)), 1)
    # The 'z' var (that we will substitute in for the function being
    # approximated) is here:
    test_case.assertEqual(len(block.component_map(Var)), 1)
    test_case.assertIsInstance(block.substitute_var, Var)
