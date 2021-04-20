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
# The parametric.mod example from sIPOPT
#
# Original implmentation by Hans Pirany is in pyomo/examples/pyomo/suffixes
#

from __future__ import print_function
from pyomo.environ import ConcreteModel, Param, Var, Objective, Constraint, NonNegativeReals, value
from pyomo.contrib.sensitivity_toolbox.sens import sipopt

def create_model():
    ''' Create a concrete Pyomo model for this example
    '''
    m = ConcreteModel()
    
    m.x1 = Var(initialize = 0.15, within=NonNegativeReals)
    m.x2 = Var(initialize = 0.15, within=NonNegativeReals)
    m.x3 = Var(initialize = 0.0, within=NonNegativeReals)
    
    m.eta1 = Param(initialize=4.5,mutable=True)
    m.eta2 = Param(initialize=1.0,mutable=True)
    
    m.const1 = Constraint(expr=6*m.x1+3*m.x2+2*m.x3-m.eta1 ==0)
    m.const2 = Constraint(expr=m.eta2*m.x1+m.x2-m.x3-1 ==0)
    m.cost = Objective(expr=m.x1**2+m.x2**2+m.x3**2)
    
    return m 

def run_example(print_flag=True):
    '''
    Execute the example
    
    Arguments:
        print_flag: Toggle on/off printing
    
    Returns:
        sln_dict: Dictionary containing solution (used for automated testing)
    
    '''
    m = create_model()    

    m.perturbed_eta1 = Param(initialize = 4.0)
    m.perturbed_eta2 = Param(initialize = 1.0)


    m_sipopt = sipopt(m,[m.eta1,m.eta2],
                        [m.perturbed_eta1,m.perturbed_eta2],
                        streamSoln=True)
    

    
    if print_flag:
        print("\nOriginal parameter values:")
        print("\teta1 =",m.eta1())
        print("\teta2 =",m.eta2())
    
        print("Initial point:")
        print("\tObjective =",value(m.cost))
        print("\tx1 =",m.x1())
        print("\tx2 =",m.x2())
        print("\tx3 =",m.x3())
        
        print("Solution with the original parameter values:")
        print("\tObjective =",m_sipopt.cost())
        print("\tx1 =",m_sipopt.x1())
        print("\tx2 =",m_sipopt.x2())
        print("\tx3 =",m_sipopt.x3())
    
        print("\nNew parameter values:")
        print("\teta1 =",m_sipopt.perturbed_eta1())
        print("\teta2 =",m_sipopt.perturbed_eta2())
    
    # This highlights one limitation of sipopt. It will only return the
    # perturbed solution. The user needs to calculate relevant values such as
    # the objective or expressions
    x1 = m_sipopt.sens_sol_state_1[m_sipopt.x1]
    x2 = m_sipopt.sens_sol_state_1[m_sipopt.x2]
    x3 = m_sipopt.sens_sol_state_1[m_sipopt.x3]
    obj = x1**2 + x2**2 + x3**2
    
    if print_flag:
        print("(Approximate) solution with the new parameter values:")
        print("\tObjective =",obj)
        print("\tx1 =",m_sipopt.sens_sol_state_1[m_sipopt.x1])
        print("\tx2 =",m_sipopt.sens_sol_state_1[m_sipopt.x2])
        print("\tx3 =",m_sipopt.sens_sol_state_1[m_sipopt.x3])
    
    # Save the results in a dictionary.
    # This is optional and makes automated testing convenient.
    # This code is not important for a Minimum Working Example (MWE) of sipopt
    d = dict()
    d['eta1'] = m.eta1()
    d['eta2'] = m.eta2()
    d['x1_init'] = m.x1()
    d['x2_init'] = m.x2()
    d['x3_init'] = m.x3()
    d['x1_sln'] = m_sipopt.x1()
    d['x2_sln'] = m_sipopt.x2()
    d['x3_sln'] = m_sipopt.x3()
    d['cost_sln'] = m_sipopt.cost()
    d['eta1_pert'] = m_sipopt.perturbed_eta1()
    d['eta2_pert'] = m_sipopt.perturbed_eta2()
    d['x1_pert'] = x1
    d['x2_pert'] = x2
    d['x3_pert'] = x3
    d['cost_pert'] = obj
    
    return d


if __name__=='__main__':
    d = run_example()
