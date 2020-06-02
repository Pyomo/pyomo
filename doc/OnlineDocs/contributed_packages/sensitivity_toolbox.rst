Sensitivity Toolbox
===================

The sensitivity toolbox provides a Pyomo interface to sIPOPT to very quickly computer approximate solutions to nonlinear programs with a small perturbation in model parameters. See the `sIPOPT documentation <https://projects.coin-or.org/Ipopt/wiki/sIpopt>`_ or the `following paper <https://link.springer.com/article/10.1007/s12532-012-0043-2>`_ for additional details:

  H. Pirnay, R. Lopez-Negrete, and L.T. Biegler, Optimal Sensitivity based on IPOPT, Math. Prog. Comp., 4(4):307--331, 2012. 


Using the Sensitivity Toolbox
-----------------------------

We will start with a motivating example:

.. math::
  \begin{align*}
  \min_{x_1,x_2,x_3} \quad & x_1^2 + x_2^2 + x_3^2 \\
  \mathrm{s.t.} \qquad & 6 x_1 + 3 x_2 + 2 x_3 - p_1 = 0 \\
  & p_2 x_1 + x_2 - x_3 - 1 = 0 \\
  & x_1, x_2, x_3 \geq 0
  \end{align*}

Here :math:`x_1`, :math:`x_2`, and :math:`x_3` are the decision variables while :math:`p_1` and :math:`p_2` are parameters. At first, let's consider :math:`p_1 = 4.5` and :math:`p_2 = 1.0`. Below is the model implemented in Pyomo.

.. doctest::
    
    # Create a concrete model
    m = ConcreteModel()
    
    # Define the variables with bounds and initial values
    m.x1 = Var(initialize = 0.15, within=NonNegativeReals)
    m.x2 = Var(initialize = 0.15, within=NonNegativeReals)
    m.x3 = Var(initialize = 0.0, within=NonNegativeReals)
    
    # Define the parameters
    m.eta1 = Param(initialize=4.5,mutable=True)
    m.eta2 = Param(initialize=1.0,mutable=True)
    
    # Define the constraints and objective
    m.const1 = Constraint(expr=6*m.x1+3*m.x2+2*m.x3-m.eta1 ==0)
    m.const2 = Constraint(expr=m.eta2*m.x1+m.x2-m.x3-1 ==0)
    m.cost = Objective(expr=m.x1**2+m.x2**2+m.x3**2)
        

The solution of this optimization problem is :math:`x_1^* = 0.15`, :math:`x_2^* = 0.15`, and :math:`x_3^* = 0.0`. But what if we change the parameter values to :math:`\hat{p}_1 = 4.0` and :math:`\hat{p}_2 = 1.0`? Is there a quick way to approximate the new solution :math:`\hat{x}_1^*`, :math:`\hat{x}_2^*`, and :math:`\hat{x}_3^*`? Yes! This is the main functionality of sIPOPT.

Next we define the perturbed parameter values :math:`\hat{p}_1` and :math:`\hat{p}_2`:

.. doctest::
    
    m.perturbed_eta1 = Param(initialize = 4.0)
    m.perturbed_eta2 = Param(initialize = 1.0)

And finally we call sIPOPT:

.. doctest::
    
    m_sipopt = sipopt(m,[m.eta1,m.eta2],
                        [m.perturbed_eta1,m.perturbed_eta2],
                        streamSoln=True)

The first argument is the Pyomo model. The second argument is a list of the original parameters. The third argument is a list of the perturbed parameters. sIPOPT requires these two lists are the same length.

We then get this output:

.. doctest::
    Ipopt 3.13.2: run_sens=yes
    
    ******************************************************************************
    This program contains Ipopt, a library for large-scale nonlinear optimization.
     Ipopt is released as open source code under the Eclipse Public License (EPL).
             For more information visit http://projects.coin-or.org/Ipopt
    ******************************************************************************
    
    This is Ipopt version 3.13.2, running with linear solver ma27.
    
    Number of nonzeros in equality constraint Jacobian...:       10
    Number of nonzeros in inequality constraint Jacobian.:        0
    Number of nonzeros in Lagrangian Hessian.............:        4
    
    Total number of variables............................:        5
                         variables with only lower bounds:        3
                    variables with lower and upper bounds:        0
                         variables with only upper bounds:        0
    Total number of equality constraints.................:        4
    Total number of inequality constraints...............:        0
            inequality constraints with only lower bounds:        0
       inequality constraints with lower and upper bounds:        0
            inequality constraints with only upper bounds:        0
            
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
       0  4.5100000e-02 4.50e+00 4.07e-01  -1.0 0.00e+00    -  0.00e+00 0.00e+00   0
       1  8.9566717e-01 1.32e-01 6.64e+00  -1.0 4.50e+00    -  1.85e-01 1.00e+00h  1
       2  5.8493209e-01 8.88e-16 5.62e-01  -1.0 2.59e-01    -  9.90e-01 1.00e+00f  1
       3  5.2819977e-01 8.88e-16 1.29e-02  -1.7 8.27e-02    -  1.00e+00 1.00e+00f  1
       4  5.0741726e-01 8.88e-16 4.99e-03  -2.5 4.19e-02    -  1.00e+00 1.00e+00f  1
       5  5.0084750e-01 8.88e-16 7.29e-04  -3.8 1.62e-02    -  1.00e+00 1.00e+00f  1
       6  5.0001574e-01 8.88e-16 1.47e-05  -5.7 2.20e-03    -  1.00e+00 1.00e+00f  1
       7  5.0000000e-01 8.88e-16 5.54e-09  -8.6 4.20e-05    -  1.00e+00 1.00e+00f  1
       
    Number of Iterations....: 7
    
                                       (scaled)                 (unscaled)
    Objective...............:   4.9999999772520909e-01    4.9999999772520909e-01
    Dual infeasibility......:   5.5407590737743817e-09    5.5407590737743817e-09
    Constraint violation....:   8.8817841970012523e-16    8.8817841970012523e-16
    Complementarity.........:   7.7252085982569584e-09    7.7252085982569584e-09
    Overall NLP error.......:   7.7252085982569584e-09    7.7252085982569584e-09
    
    
    Number of objective function evaluations             = 8
    Number of objective gradient evaluations             = 8
    Number of equality constraint evaluations            = 8
    Number of inequality constraint evaluations          = 0
    Number of equality constraint Jacobian evaluations   = 8
    Number of inequality constraint Jacobian evaluations = 0
    Number of Lagrangian Hessian evaluations             = 7
    Total CPU secs in IPOPT (w/o function evaluations)   =      0.002
    Total CPU secs in NLP function evaluations           =      0.000
    
    EXIT: Optimal Solution Found.

We can now inspect the solution :math:`x_1^*`, :math:`x_2^*`, and :math:`x_3^*`:

.. doctest::
    
    print("\nOriginal parameter values:")
    print("\teta1 =",m.eta1())
    print("\teta2 =",m.eta2())
    
    print("Solution with the original parameter values:")
    print("\tx1 =",m.x1())
    print("\tx2 =",m.x2())
    print("\tx3 =",m.x3())

Which gives the output:

.. doctest::
    
    Original parameter values:
        eta1 = 4.5
        eta2 = 1.0
    Solution with the original parameter values:
        x1 = 0.15
        x2 = 0.15
        x3 = 0.0

Likewise, we can inspect the approximate solution :math:`\hat{x}_1^*`, :math:`\hat{x}_2^*`, and :math:`\hat{x}_3^*`:

.. doctest::
    
    print("\nNew parameter values:")
    print("\teta1 =",m_sipopt.perturbed_eta1())
    print("\teta2 =",m_sipopt.perturbed_eta2())
    
    print("(Approximate) solution with the new parameter values:")
    print("\tx1 =",m_sipopt.x1())
    print("\tx2 =",m_sipopt.x2())
    print("\tx3 =",m_sipopt.x3())

Which gives the output:

.. doctest::
    
    New parameter values:
        eta1 = 4.0
        eta2 = 1.0
    (Approximate) solution with the new parameter values:
        x1 = 0.5000000037913185
        x2 = 0.4999999939338906
        x3 = 0.0

Installing sIPOPT
-----------------

TODO: Fill in with details.

Sensitivity Toolbox Interface
-----------------------------

.. autoclass:: pyomo.contrib.sensitivity_toolbox.sens.sipopt
    :members: