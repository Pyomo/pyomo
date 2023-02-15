#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#
#  Pyomo.DoE was produced under the Department of Energy Carbon Capture Simulation 
#  Initiative (CCSI), and is copyright (c) 2022 by the software owners: 
#  TRIAD National Security, LLC., Lawrence Livermore National Security, LLC., 
#  Lawrence Berkeley National Laboratory, Pacific Northwest National Laboratory,  
#  Battelle Memorial Institute, University of Notre Dame,
#  The University of Pittsburgh, The University of Texas at Austin, 
#  University of Toledo, West Virginia University, et al. All rights reserved.
# 
#  NOTICE. This Software was developed under funding from the 
#  U.S. Department of Energy and the U.S. Government consequently retains 
#  certain rights. As such, the U.S. Government has been granted for itself
#  and others acting on its behalf a paid-up, nonexclusive, irrevocable, 
#  worldwide license in the Software to reproduce, distribute copies to the 
#  public, prepare derivative works, and perform publicly and display
#  publicly, and to permit other to do so.
#  ___________________________________________________________________________


class SpecialSet: 
    def __init__(self):
        """This class defines variable names with provided names and indexes.
        """
        self.special_set = []

    def specify(self, self_define_res):
        """
        Used for user to provide defined string names

        Parameter
        ---------
        self_define_res: a ``list`` of ``string``, containing the measurement variable names with indexs, 
            for e.g. "C['CA', 23, 0]". 
            If this is defined, no need to define ``measurement_var``, ``extra_idx``, ``time_idx``.
        """
        self.special_set = self_define_res

    def add_elements(self, var_name, extra_index=None, time_index=[0]):
        """
        Used for generating string names with indexes. 

        Parameter 
        ---------
        var_name: a ``list`` of measurement var names 
        extra_index: a ``list`` containing extra indexes except for time indexes 
            if default (None), no extra indexes needed for all var in var_name
            if it is a ``list`` of strings or int or floats, it is a set of one index, for every var in var_name 
            if it is a ``list`` of lists: they are multiple sets of indexes, for every var in var_name 
            if it is a ``list`` of ``list`` of ``list``, they are different multiple sets of indexes for different var_name 
        time_index: a ``list`` containing time indexes
            default choice is [0], means this is an algebraic variable 
            if it is a ``list`` of integers or floats, it is the time set for every var in var_name 
            if it is a ``list`` of ``lists``, they are different time set for different var in var_name
        """

        