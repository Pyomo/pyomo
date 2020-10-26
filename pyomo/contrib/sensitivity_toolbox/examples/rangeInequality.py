#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#

from pyomo.environ import ConcreteModel, Param, Var, Constraint, inequality

from pyomo.contrib.sensitivity_toolbox.sens import sipopt


def create_model():

    m = ConcreteModel()

    m.a = Param(initialize=0, mutable=True)
    m.b = Param(initialize=1, mutable=True)


    m.x = Var(initialize = 1.0)        
    m.y = Var()
    m.C_rangedIn = Constraint(expr=inequality(m.a,m.x,m.b))
    m.C_equal = Constraint(expr=m.y==m.b)
    m.C_singleBnd = Constraint(expr=m.x<=m.b)


    return m

if __name__=='__main__':
    m = create_model()

    m.pert_a = Param(initialize=0.01)
    m.pert_b = Param(initialize=1.01)


    m_sipopt = sipopt(m,[m.a,m.b],[m.pert_a,m.pert_b],
                      streamSoln=True)
