# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 13:07:51 2023

@author: jlgearh
"""

from obbt import obbt_analysis
from tests.test_cases import get_continuous_prob_1

m = get_continuous_prob_1()
results, solutions = obbt_analysis(m, warmstart=True, solver='cplex')