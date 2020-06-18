Sensitivity Toolbox
===================

The sensitivity toolbox provides a Pyomo interface to sIPOPT to very quickly compute approximate solutions to nonlinear programs with a small perturbation in model parameters. See the `sIPOPT documentation <https://projects.coin-or.org/Ipopt/wiki/sIpopt>`_ or the `following paper <https://link.springer.com/article/10.1007/s12532-012-0043-2>`_ for additional details:

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

    Import Pyomo and sipopt from the sensitivity toolbox
    >>> from pyomo.environ import * 
    >>> from pyomo.contrib.sensitivity_toolbox.sens import sipopt
    
    Create a concrete model
    >>> m = ConcreteModel()
    
    Define the variables with bounds and initial values
    >>> m.x1 = Var(initialize = 0.15, within=NonNegativeReals)
    >>> m.x2 = Var(initialize = 0.15, within=NonNegativeReals)
    >>> m.x3 = Var(initialize = 0.0, within=NonNegativeReals)
    
    Define the parameters
    >>> m.eta1 = Param(initialize=4.5,mutable=True)
    >>> m.eta2 = Param(initialize=1.0,mutable=True)
    
    Define the constraints and objective
    >>> m.const1 = Constraint(expr=6*m.x1+3*m.x2+2*m.x3-m.eta1 ==0)
    >>> m.const2 = Constraint(expr=m.eta2*m.x1+m.x2-m.x3-1 ==0)
    >>> m.cost = Objective(expr=m.x1**2+m.x2**2+m.x3**2)
        

The solution of this optimization problem is :math:`x_1^* = 0.15`, :math:`x_2^* = 0.15`, and :math:`x_3^* = 0.0`. But what if we change the parameter values to :math:`\hat{p}_1 = 4.0` and :math:`\hat{p}_2 = 1.0`? Is there a quick way to approximate the new solution :math:`\hat{x}_1^*`, :math:`\hat{x}_2^*`, and :math:`\hat{x}_3^*`? Yes! This is the main functionality of sIPOPT.

Next we define the perturbed parameter values :math:`\hat{p}_1` and :math:`\hat{p}_2`:

.. doctest::
    
    >>> m.perturbed_eta1 = Param(initialize = 4.0)
    >>> m.perturbed_eta2 = Param(initialize = 1.0)

And finally we call sIPOPT:

.. doctest::
    :skipif: not sipopt_available

    >>> m_sipopt = sipopt(m,[m.eta1,m.eta2], [m.perturbed_eta1,m.perturbed_eta2], streamSoln=True)
    Ipopt 3.13.2: run_sens=yes
    ...
    ******************************************************************************
    This program contains Ipopt, a library for large-scale nonlinear optimization.
     Ipopt is released as open source code under the Eclipse Public License (EPL).
             For more information visit http://projects.coin-or.org/Ipopt
    ******************************************************************************
    ...
    EXIT: Optimal Solution Found.

The first argument is the Pyomo model. The second argument is a list of the original parameters. The third argument is a list of the perturbed parameters. sIPOPT requires these two lists are the same length. The ```...``` represents extra lines of output that were cut from this page for brevity.

We can now inspect the solution :math:`x_1^*`, :math:`x_2^*`, and :math:`x_3^*`:

.. doctest::
    :skipif: not sipopt_available
    
    Original parameter values
    >>> print("eta1 =",m.eta1())
    eta1 = 4.5
    
    >>> print("eta2 =",m.eta2())
    eta2 = 1.0
    
    Solution with the original parameter values:
    >>> print("x1 =",m.x1())
    x1 = 0.15
    
    >>> print("x2 =",m.x2())
    x2 = 0.15
    
    >>> print("x3 =",m.x3())
    x3 = 0.0

Likewise, we can inspect the approximate solution :math:`\hat{x}_1^*`, :math:`\hat{x}_2^*`, and :math:`\hat{x}_3^*`:

.. doctest::
    :skipif: not sipopt_available    

    New parameter values:
    >>> print("eta1 =",m_sipopt.perturbed_eta1())
    eta1 = 4.0
    
    >>> print("eta2 =",m_sipopt.perturbed_eta2())
    eta2 = 1.0
    
    (Approximate) solution with the new parameter values:
    >>> print("x1 =",m_sipopt.x1())
    x1 = 0.5000000037913185
    
    >>> print("x2 =",m_sipopt.x2())
    x2 = 0.4999999939338906
    
    >>> print("x3 =",m_sipopt.x3())
    x3 = 0.0

Installing sIPOPT
-----------------

The sensitivity toolbox requires sIPOPT is installed and available in your system PATH. See the IPOPT documentation for detailed instructions:

* https://coin-or.github.io/Ipopt/INSTALL.html
* https://projects.coin-or.org/Ipopt/wiki/sIpopt
* https://coin-or.github.io/coinbrew/

.. note:: If you get an error that ``ipopt_sens`` cannot be found, you need to make sure sIPOPT was installed and that it is in the system path.

Sensitivity Toolbox Interface
-----------------------------

.. autoclass:: pyomo.contrib.sensitivity_toolbox.sens.sipopt
    :members:
