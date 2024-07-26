#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# @Import_Syntax
import pyomo.environ as aml

# @Import_Syntax

datafile = None

# @AbstractModels
m = aml.AbstractModel()
# ... define model ...
instance = m.create_instance(datafile)

# @AbstractModels
del datafile
del instance
# @ConcreteModels
m = aml.ConcreteModel()
m.b = aml.Block()
# @ConcreteModels


# @Sets_1
m.s = aml.Set(initialize=[1, 2], ordered=True)
# @Sets_1
# @Sets_2
# [1,2,3]
m.q = aml.RangeSet(1, 3)
# @Sets_2


# @Parameters_single
m.p = aml.Param(mutable=True, initialize=0)


# @Parameters_single
# @Parameters_dict
# pd[1] = 0, pd[2] = 1
def pd_(m, i):
    return m.s.ord(i) - 1


m.pd = aml.Param(m.s, mutable=True, rule=pd_)
# @Parameters_dict
# @Parameters_list

#
# No ParamList exists
#


# @Parameters_list


# @Variables_single
m.v = aml.Var(initialize=1.0, bounds=(1, 4))

# @Variables_single
# @Variables_dict
m.vd = aml.Var(m.s, bounds=(None, 9))


# @Variables_dict
# @Variables_list
# used 1-based indexing
def vl_(m, i):
    return (i, None)


m.vl = aml.VarList(bounds=vl_)
for j in m.q:
    m.vl.add()
# @Variables_list

# @Constraints_single
m.c = aml.Constraint(expr=sum(m.vd.values()) <= 9)


# @Constraints_single
# @Constraints_dict
def cd_(m, i, j):
    return m.vd[i] == j


m.cd = aml.Constraint(m.s, m.q, rule=cd_)


# @Constraints_dict
# @Constraints_list
# uses 1-based indexing
m.cl = aml.ConstraintList()
for j in m.q:
    m.cl.add(aml.inequality(-5, m.vl[j] - m.v, 5))
# @Constraints_list


# @Expressions_single
m.e = aml.Expression(expr=-m.v)


# @Expressions_single
# @Expressions_dict
def ed_(m, i):
    return -m.vd[i]


m.ed = aml.Expression(m.s, rule=ed_)
# @Expressions_dict
# @Expressions_list

#
# No ExpressionList exists
#

# @Expressions_list


# @Objectives_single
m.o = aml.Objective(expr=-m.v)


# @Objectives_single
# @Objectives_dict
def od_(m, i):
    return -m.vd[i]


m.od = aml.Objective(m.s, rule=od_)
# @Objectives_dict
# @Objectives_list
# uses 1-based indexing
m.ol = aml.ObjectiveList()
for j in m.q:
    m.ol.add(-m.vl[j])

# @Objectives_list


# @SOS_single
m.sos1 = aml.SOSConstraint(var=m.vl, level=1)
m.sos2 = aml.SOSConstraint(var=m.vd, level=2)


# @SOS_single
# @SOS_dict
def sd_(m, i):
    if i == 1:
        t = list(m.vd.values())
    elif i == 2:
        t = list(m.vl.values())
    return t


m.sd = aml.SOSConstraint([1, 2], rule=sd_, level=1)
# @SOS_dict
# @SOS_list

#
# No SOSConstraintList exists
#

# @SOS_list


# @Suffix_single
m.dual = aml.Suffix(direction=aml.Suffix.IMPORT)
# @Suffix_single
# @Suffix_dict
#
# No SuffixDict exists
#
# @Suffix_dict


# @Piecewise_1d
breakpoints = [1, 2, 3, 4]
values = [1, 2, 1, 2]
m.f = aml.Var()
m.pw = aml.Piecewise(m.f, m.v, pw_pts=breakpoints, f_rule=values, pw_constr_type='EQ')
# @Piecewise_1d


m.pprint()
