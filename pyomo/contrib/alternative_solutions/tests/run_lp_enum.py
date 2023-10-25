# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:55:46 2023

@author: jlgearh
"""

import pyomo.contrib.alternative_solutions.tests.test_cases as tc
from pyomo.contrib.alternative_solutions import lp_enum

m = tc.get_2d_degenerate_lp()
sols = lp_enum.enumerate_linear_solutions(m)