Writing Nonlinear Optimization Algorithms
=========================================

Writing nonlinear optimization algorithms is challenging. In general, the implementation of these algorithms requires a good understanding of the mathematical theory as well of software development. With PyNumero we hope to motivate students in the field of nonlinear optimization to write and test their own algorithms. We have implemented different examples for users to get familiar with PyNumero. All these examples utilize the building blocks described in the previous sections. Among the algorithms implemented in PyNumero we have

* Newton for solving nonlinear system of equations
* Mehrotras algorithm for convex QPs
* Interior-Point (basic implementation of Ipopts algorithm)
* Penalty-Interior-Point (basic implementation of WORHPs algorithm from Renke Kuhlmann)

Another good tool for implementing nonlinear optimization algorithms in PyNumero is the **NLPState** interface. The following is an example of the output of the interior-point algorithm implementation in PyNumero::

  ********************************************************************************************
  ********************** THIS ALGORITHM WAS WRITTEN IN PYNUMERO ******************************
  ********************************************************************************************
  ********************************************************************************************

  ALGORITHM NAME: Interior-Point

  LINEAR SOLVER NAME: ma27

  PROBLEM STATISTICS:

  Number of nonzeros in equality constraint Jacobian...:        2
  Number of nonzeros in inequality constraint Jacobian.:        2
  Number of nonzeros in Lagrangian Hessian.............:        4
  Number of primal variables...........................:        3
  Variables with only lower bounds:                             2
  Variables with lower and upper bounds:                        0
  Variables with only upper bounds:                             0
  Total number of equality constraints.................:        1
  Total number of inequality constraints...............:        1
  Inequality constraints with only lower bounds:                0
  Inequality constraints with lower and upper bounds:           0
  Inequality constraints with only upper bounds:                0

  ITERATIONS:

  iter    objective    inf_pr   inf_du   inf_cmp  lg(mu)  ||d||   lg(rg) alpha_du alpha_pr  ls
    0 -5.0400000e+02 5.00e+00 5.77e+02  3.90e+00  -1.0  0.00e+00   --   1.00e+00 1.00e+00   0
    1 -5.1540864e+02 5.00e+00 5.76e+02  1.70e+00  -1.0  1.78e-01   --   1.39e-01 8.43e-04   1
    2 -4.2746518e+02 5.12e-01 2.92e+01  2.49e-01  -1.0  7.30e-01   --   1.00e+00 1.00e+00   1
    3 -4.2854826e+02 1.32e-03 1.10e+00  1.20e-02  -1.0  1.70e-01   --   1.00e+00 1.00e+00   1
    4 -4.2853628e+02 4.03e-07 1.94e-03  9.72e-02  -2.5  7.26e-03   --   1.00e+00 1.00e+00   1
    5 -4.2853624e+02 1.83e-12 5.24e+01  1.04e-06  -2.5  1.31e-05   --   1.00e+00 1.00e+00   1
    6 -4.2863595e+02 4.07e-10 2.08e-01  2.54e-03  -2.5  1.85e-03   --   1.00e+00 2.89e-02   1
    7 -4.2863342e+02 4.05e-10 1.31e-06  2.68e-03  -3.8  1.88e-04   --   1.00e+00 1.00e+00   1
    8 -4.2863341e+02 0.00e+00 5.12e+01  9.56e-06  -3.8  1.87e-07   --   1.00e+00 1.00e+00   1
    9 -4.2863624e+02 3.13e-13 3.09e-03  1.50e-04  -3.8  5.25e-05   --   1.00e+00 5.26e-02   1
  iter    objective    inf_pr   inf_du   inf_cmp  lg(mu)  ||d||   lg(rg) alpha_du alpha_pr  ls
   10 -4.2863609e+02 3.13e-13 1.01e-09  1.49e-04  -5.7  5.22e-06   --   1.00e+00 1.00e+00   1
   11 -4.2863609e+02 3.55e-15 5.32e+01  8.96e-09  -5.7  1.68e-10   --   1.00e+00 1.00e+00   1
   12 -4.2863624e+02 7.11e-15 7.58e-04  1.84e-06  -5.7  2.79e-06   --   1.00e+00 1.24e-02   1
   13 -4.2863623e+02 3.55e-15 3.97e-11  1.84e-06  -8.6  2.89e-07   --   1.00e+00 1.00e+00   1
   14 -4.2863623e+02 0.00e+00 5.38e+01  2.63e-11  -8.6  4.87e-13   --   1.00e+00 1.00e+00   1
   15 -4.2863624e+02 3.55e-15 8.72e-05  2.45e-09  -8.6  3.42e-08   --   1.00e+00 1.35e-03   1
    
  SOLUTION STATISTICS:
    
  Number of Iterations...............................:          16
  Objective..........................................:          -4.2863623583904507e+02
  Dual infeasibility.................................:          1.2079226507921703e-13
  Constraint violation...............................:          0.0000000000000000e+00
  Complementarity....................................:          1.5059035596800621e-09
