#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyomo.environ as pyo
import numpy as np
from pyomo.contrib.sensitivity_toolbox.sens import get_dsdp, get_dfds_dcds


# In[2]:


def create_model():
    ### Create optimization model
    m = pyo.ConcreteModel()
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # Define variables
    m.x1 = pyo.Var(bounds=(0,10))
    m.x2 = pyo.Var(bounds=(0,10))
    m.x3 = pyo.Var(bounds=(0,10))

    # Define parameters
    #m.p1 = pyo.Param(initialize=10, mutable=True)
    #m.p2 = pyo.Param(initialize=5, mutable=True)
    # Important Tip: The uncertain parameters need to be defined at Pyomo variables
    m.p1 = pyo.Var(initialize=10)
    m.p2 = pyo.Var(initialize=5)
    # Important Tip: We fix these to first solve with Ipopt. We unfix below
    # before using the uncertainty propagation toolbox.
    m.p1.setlb(10)
    m.p1.setub(10)
    m.p2.setlb(5)
    m.p2.setub(5)

    # Define constraints
    m.con1 = pyo.Constraint(expr=m.x1 + m.x2-m.p1==0)
    m.con2 = pyo.Constraint(expr=m.x2 + m.x3-m.p2==0)

    # Define objective
    m.obj = pyo.Objective(expr=m.p1*m.x1+ m.p2*(m.x2**2) + m.p1*m.p2, sense=pyo.minimize)
    
    return m 


# In[14]:


if __name__ == "__main__":
    ### Analytic solution
    '''
    At the optimal solution, none of the bounds are active. As long as the active set
    does not change (i.e., none of the bounds become active), the
    first order optimality conditions reduce to a simple linear system.
    '''

    m = create_model()

    # dual variables (multipliers)
    v2_ = 0
    v1_ = m.p1()

    # primal variables
    x2_ = (v1_ + v2_)/(2 * m.p2())
    x1_ = m.p1() - x2_
    x3_ = m.p2() - x2_

    print("\nAnalytic solution:")
    print("x1 =",x1_)
    print("x2 =",x2_)
    print("x3 =",x3_)
    print("v1 =",v1_)
    print("v2 =",v2_)

    ### Analytic sensitivity
    '''
    Using the analytic solution above, we can compute the sensitivies of x and v to
    perturbations in p1 and p2.
    The matrix dx_dp constains the sensitivities of x to perturbations in p
    ''' 

    # Initialize sensitivity matrix Nx x Np
    # Rows: variables x
    # Columns: parameters p
    dx_dp = np.zeros((3,2))

    # dx2/dp1 = 1/(2 * p2)
    dx_dp[1, 0] = 1/(2*m.p2())

    # dx2/dp2 = -(v1 + v2)/(2 * p2**2)
    dx_dp[1,1] = -(v1_ + v2_)/(2 * m.p2()**2)

    # dx1/dp1 = 1 - dx2/dp1
    dx_dp[0, 0] = 1 - dx_dp[1,0]

    # dx1/dp2 = 0 - dx2/dp2
    dx_dp[0, 1] = 0 - dx_dp[1,1]

    # dx3/dp1 = 1 - dx2/dp1
    dx_dp[2, 0] = 0 - dx_dp[1,0]

    # dx3/dp2 = 0 - dx2/dp2
    dx_dp[2, 1] = 1 - dx_dp[1,1]

    print("\n\ndx/dp =\n",dx_dp)


    '''
    Similarly, we can compute the gradients df_dx, df_dp
    and Jacobians dc_dx, dc_dp
    '''

    # Initialize 1 x 3 array to store (\partial f)/(\partial x)
    # Elements: variables x
    df_dx = np.zeros(3)

    # df/dx1 = p1
    df_dx[0] = m.p1()

    # df/dx2 = p2
    df_dx[1] = 2 * m.p2() * x2_

    # df/dx3 = 0

    print("\n\ndf/dx =\n",df_dx)

    # Initialize 1 x 2 array to store (\partial f)/(\partial p)
    # Elements: parameters p
    df_dp = np.zeros(2)

    # df/dxp1 = x1 + p2
    df_dp[0] = x1_ + m.p2()

    # df/dp2 = x2**2 + p1
    df_dp[1] = x2_**2 + m.p1()

    print("\n\ndf/dp =\n",df_dp)

    # Initialize 2 x 3 array to store (\partial c)/(\partial x)
    # Rows: constraints c
    # Columns: variables x
    dc_dx = np.zeros((2,3))

    # dc1/dx1 = 1
    dc_dx[0,0] = 1

    # dc1/dx2 = 1
    dc_dx[0,1] = 1

    # dc2/dx2 = 1
    dc_dx[1,1] = 1

    # dc2/dx3 = 1
    dc_dx[1,2] = 1

    # Remaining entries are 0

    print("\n\ndc/dx =\n",dc_dx)

    # Initialize 2 x 2 array to store (\partial c)/(\partial x)
    # Rows: constraints c
    # Columns: variables x
    dc_dp = np.zeros((2,2))

    # dc1/dp1 = -1
    dc_dp[0,0] = -1

    # dc2/dp2 = -1
    dc_dp[1,1] = -1

    # Remaining entries are 0

    print("\n\ndc/dp =\n",dc_dp)
    
    print('======= Uncertainty propagation kaug solution ======== ')
    
    variable_name = ['p1','p2']
    
    # get df/dx, df/dp, dc/dx, dc/dp
    gradient_f, gradient_c, col,row, line_dic = get_dfds_dcds(m, variable_name, tee=False)
    
    # get ds/dp
    dsdp_re, col = get_dsdp(m, variable_name, {'p1':10, 'p2':5}, tee=False)

    # check ds/dp 
    print('\n\nds/dp=\n',)
    var_idx = np.array([True,True,False,False,True])
    dsdp_array = (dsdp_re.toarray().T)[var_idx,:]
    
    print(dsdp_array)
    
    print('\n\ndf/dp, df/dx=\n',)
    print(gradient_f)
    
    print('\n\ndc/dp, dc/dx=\n',)
    print(gradient_c)
    print('***Lacks elements in dc/dx***')


# In[ ]:




