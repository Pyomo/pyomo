Pyomo.DoE
=========

.. figure:: ../../../logos/doe/PyomoDoE-md.png
   :scale: 25 %
   :align: right

**Pyomo.DoE** (Pyomo Design of Experiments) is a Python library for model-based design of experiments using science-based models.



Pyomo.DoE was developed by **Jialu Wang** and **Alexander Dowling** at the University of Notre Dame as part of the `Carbon Capture Simulation for Industry Impact (CCSI2) <https://www.acceleratecarboncapture.org/>`_
project, funded through the U.S. Department of Energy Office of Fossil Energy and Carbon Management. Special thank you to John Siirola and Bethany Nicholson for extensive code reviews, suggestions, and improvements to Pyomo.DoE.

If you use Pyomo.DoE, please cite:

Jialu Wang and Alexander W. Dowling (2022). "Pyomo.DoE: An open‐source package for model‐based design of experiments in Python." 
*AIChE Journal* 68(12), e17813. `doi: 10.1002/aic.17813 <https://doi.org/10.1002/aic.17813>`_



Methodology Overview
---------------------

Model-based Design of Experiments (MBDoE) is a technique to maximize the information gain of experiments by directly using science-based models with physically meaningful parameters. It is one key component in the model calibration and uncertainty quantification workflow shown below:

.. figure:: flowchart.png
   :width: 99 %
   :align: center

   Pyomo.DoE integrates exploratory analysis, parameter estimation, uncertainty analysis, and MBDoE into an iterative framework to select, refine, and calibrate science-based mathematical models with quantified uncertainty. Currently, Pyomo.DoE focused on increasing parameter precision.

Pyomo.DoE provides science-based MBDoE capabilities to the Pyomo ecosystem. The user provides one Pyomo model, a set of parameter nominal values,
the allowable design spaces for design variables, and the assumed observation error structure (e.g., covariance matrix).
During exploratory analysis, Pyomo.DoE checks if the model parameters can be inferred from the proposed measurements or preliminary data.
Pyomo.DoE then recommends optimized experimental conditions for collecting more data.
Parameter estimation via `Parmest <https://pyomo.readthedocs.io/en/stable/contributed_packages/parmest/index.html>`_ can estimate uncertainty parameters from data and compute a parameter covariance matrix.
If the parameter uncertainties are sufficiently small, the workflow terminates and returns the final model with quantified parametric uncertainty.
Otherwise, Pyomo.DoE recommends the best next set of experimental conditions to generate new data.

Below is an overview of the type of optimization models Pyomo.DoE can accomodate:

* Pyomo.DoE is suitable for optimization models of **continuous** variables
* Pyomo.DoE can handle **equality constraints** defining state variables
* Pyomo.DoE supports (Partial) Differential-Algebraic Equations (PDAE) models via `Pyomo.DAE <https://pyomo.readthedocs.io/en/stable/modeling_extensions/dae.html>`_
* Pyomo.DoE also supports models with only algebraic constraints

Pyomo.DoE considers the following DAE model:

.. math::
    \begin{align*}
        & \dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), \mathbf{z}(t), \mathbf{y}(t), \mathbf{u}(t), \overline{\mathbf{w}}, \boldsymbol{\theta}) \\
        & \mathbf{g}(\mathbf{x}(t),  \mathbf{z}(t), \mathbf{y}(t), \mathbf{u}(t), \overline{\mathbf{w}},\boldsymbol{\theta})=\mathbf{0} \\
        & \mathbf{y} =\mathbf{h}(\mathbf{x}(t), \mathbf{z}(t), \mathbf{u}(t), \overline{\mathbf{w}},\boldsymbol{\theta}) \\
        & \mathbf{f}^{\mathbf{0}}\left(\dot{\mathbf{x}}\left(t_{0}\right), \mathbf{x}\left(t_{0}\right), \mathbf{z}(t_0), \mathbf{y}(t_0), \mathbf{u}\left(t_{0}\right), \overline{\mathbf{w}}, \boldsymbol{\theta})\right)=\mathbf{0} \\
        & \mathbf{g}^{\mathbf{0}}\left( \mathbf{x}\left(t_{0}\right),\mathbf{z}(t_0), \mathbf{y}(t_0), \mathbf{u}\left(t_{0}\right), \overline{\mathbf{w}}, \boldsymbol{\theta}\right)=\mathbf{0}\\
        &\mathbf{y}^{\mathbf{0}}\left(t_{0}\right)=\mathbf{h}\left(\mathbf{x}\left(t_{0}\right),\mathbf{z}(t_0), \mathbf{u}\left(t_{0}\right), \overline{\mathbf{w}}, \boldsymbol{\theta}\right)
    \end{align*}

where:

*  :math:`\boldsymbol{\theta} \in \mathbb{R}^{N_p}` are unknown model parameters.
*  :math:`\mathbf{x} \subseteq \mathcal{X}` are dynamic state variables which characterize trajectory of the system, :math:`\mathcal{X} \in \mathbb{R}^{N_x \times N_t}`.
*  :math:`\mathbf{z} \subseteq \mathcal{Z}` are algebraic state variables, :math:`\mathcal{Z} \in \mathbb{R}^{N_z \times N_t}`.
*  :math:`\mathbf{u} \subseteq \mathcal{U}` are time-varying decision variables,  :math:`\mathcal{U} \in \mathbb{R}^{N_u \times N_t}`.
*  :math:`\overline{\mathbf{w}} \in \mathbb{R}^{N_w}` are time-invariant decision variables.
*  :math:`\mathbf{y} \subseteq \mathcal{Y}` are measurement response variables,   :math:`\mathcal{Y} \in \mathbb{R}^{N_r \times N_t}`.
*  :math:`\mathbf{f}(\cdot)` are differential equations.
*  :math:`\mathbf{g}(\cdot)` are algebraic equations.
*  :math:`\mathbf{h}(\cdot)` are measurement functions.
*  :math:`\mathbf{t} \in \mathbb{R}^{N_t \times 1}` is a union of all time sets.

.. note::
    * Process models provided to Pyomo.DoE should define an extra scenario index for all state variables and all parameters, as the first index before any other index. The next version of Pyomo.DoE will remove this requirement.
    * Process models must include an index for time, named ``t``. For steady-state models, ``t`` should be ``[0]``.
    * Measurements can have an extra index (e.g., spatial domain) besides time.
    * Parameters and design variables should be defined as Pyomo ``var`` components on the model to use ``direct_kaug`` mode. Other modes allow these to be defines as Pyomo ``Param`` objects.
    * Create model function should take scenarios as the first argument of this function.
    * Design variables are defined with and only with a time index.

Pyomo.DoE solves the following DAE-constrainted optimizaiton problem:

.. math::
    \begin{equation}
    \begin{aligned}
        \underset{\boldsymbol{\varphi}}{\max} \quad & \Psi (\mathbf{M}(\mathbf{\hat{y}}, \boldsymbol{\varphi})) \\
        \text{s.t.} \quad & \mathbf{M}(\boldsymbol{\hat{\theta}}, \boldsymbol{\varphi}) = \sum_r^{N_r} \sum_{r'}^{N_r} \tilde{\sigma}_{(r,r')}\mathbf{Q}_r^\mathbf{T} \mathbf{Q}_{r'} + \mathbf{V}^{-1}_{\boldsymbol{\theta}}(\boldsymbol{\hat{\theta}}) \\
        & \dot{\mathbf{x}}(t) = \mathbf{f}(\mathbf{x}(t), \mathbf{z}(t), \mathbf{y}(t), \mathbf{u}(t), \overline{\mathbf{w}}, \boldsymbol{\theta}) \\
        & \mathbf{g}(\mathbf{x}(t),  \mathbf{z}(t), \mathbf{y}(t), \mathbf{u}(t), \overline{\mathbf{w}},\boldsymbol{\theta})=\mathbf{0} \\
        & \mathbf{y} =\mathbf{h}(\mathbf{x}(t), \mathbf{z}(t), \mathbf{u}(t), \overline{\mathbf{w}},\boldsymbol{\theta}) \\
        & \mathbf{f}^{\mathbf{0}}\left(\dot{\mathbf{x}}\left(t_{0}\right), \mathbf{x}\left(t_{0}\right), \mathbf{z}(t_0), \mathbf{y}(t_0), \mathbf{u}\left(t_{0}\right), \overline{\mathbf{w}}, \boldsymbol{\theta})\right)=\mathbf{0} \\
        & \mathbf{g}^{\mathbf{0}}\left( \mathbf{x}\left(t_{0}\right),\mathbf{z}(t_0), \mathbf{y}(t_0), \mathbf{u}\left(t_{0}\right), \overline{\mathbf{w}}, \boldsymbol{\theta}\right)=\mathbf{0}\\
        &\mathbf{y}^{\mathbf{0}}\left(t_{0}\right)=\mathbf{h}\left(\mathbf{x}\left(t_{0}\right),\mathbf{z}(t_0), \mathbf{u}\left(t_{0}\right), \overline{\mathbf{w}}, \boldsymbol{\theta}\right)
    \end{aligned}
    \end{equation}

where:

*  :math:`\boldsymbol{\varphi}` are design variables, which are manipulated to maximize the information content of experiments. It should consist of one or more of  :math:`\mathbf{u}(t), \mathbf{y}^{\mathbf{0}}({t_0}),\overline{\mathbf{w}}`. With a proper model formulation, the timepoints for control or measurements :math:`\mathbf{t}` can also be degrees of freedom.
*  :math:`\mathbf{M}` is the Fisher information matrix (FIM), estimated as the inverse of the covariance matrix of parameter estimates  :math:`\boldsymbol{\hat{\theta}}`. A large FIM indicates more information contained in the experiment for parameter estimation.
*  :math:`\mathbf{Q}` is the dynamic sensitivity matrix, containing the partial derivatives of  :math:`\mathbf{y}` with respect to  :math:`\boldsymbol{\theta}`.
*  :math:`\Psi(\cdot)` is the design criteria computed from the FIM.
*  :math:`\mathbf{V}_{\boldsymbol{\theta}}(\boldsymbol{\hat{\theta}})^{-1}` is the FIM of previous experiments.

Pyomo.DoE provides four design criteria  :math:`\Psi(\cdot)` to measure the size of FIM:

.. list-table:: Pyomo.DoE design criteria
    :header-rows: 1
    :class: tight-table

    * - Design criterion
      - Computation
      - Geometrical meaning
    * - A-optimality
      -   :math:`\text{trace}({\mathbf{M}})`
      - Dimensions of the enclosing box of the confidence ellipse
    * - D-optimality
      -   :math:`\text{det}({\mathbf{M}})`
      - Volume of the confidence ellipse
    * - E-optimality
      -   :math:`\text{min eig}({\mathbf{M}})`
      - Size of the longest axis of the confidence ellipse
    * - Modified E-optimality
      -   :math:`\text{cond}({\mathbf{M}})`
      - Ratio of the longest axis to the shortest axis of the confidence ellipse

In order to solve problems of the above, Pyomo.DoE implements the 2-stage stochastic program. See `Wang and Dowling (2022) <https://doi.org/10.1002/aic.17813>`_ for details.

Pyomo.DoE Required Inputs
--------------------------------
Pyomo.DoE requires the following inputs:

* A Python function that creates the process model. This is similar to interface for `Parmest <https://pyomo.readthedocs.io/en/stable/contributed_packages/parmest/index.html>`_.
* Dictionary of parameters and their nominal value
* Dictionary of measurements and their measurement time points
* Dictionary of design variables and their control time points
* A Numpy ``array`` containing the prior FIM
* Local and global nonlinear optimization solver object

Below is a list of arguments that Pyomo.DoE expects the user to provide.

param_init : ``dictionary``
    A ``dictionary`` of parameter names and values. If they are an indexed variable, put the variable name and index in a nested ``Dictionary``.

design_variable_timepoints : ``dictionary``
    A ``dictionary`` of design variable names and its control time points. If this design var is independent of time (constant), set the time to [0]

measurement_object : ``object``
    An ``object`` of the measurements, provided by the measurement class.


create_model : ``function``
    A ``function`` returning a deterministic process model.


prior_FIM : ``array``
    An ``array`` defining the Fisher information matrix (FIM) for prior experiments, default is a zero matrix.

Pyomo.DoE Solver Interface
---------------------------

.. figure:: uml.png
   :scale: 25 %


.. autoclass:: pyomo.contrib.doe.doe.DesignOfExperiments
    :members: __init__, stochastic_program, compute_FIM, run_grid_search

.. Note::
    ``stochastic_program()`` includes the following steps:
        #.  Build two-stage stochastic programming optimization model where scenarios correspond to finite difference approximations for the Jacobian of the response variables with respect to calibrated model parameters
        #.  Fix the experiment design decisions and solve a square (i.e., zero degrees of freedom) instance of the two-stage DoE problem. This step is for initialization.
        #.  Unfix the experiment design decisions and solve the two-stage DoE problem.

.. autoclass:: pyomo.contrib.doe.measurements.Measurements
    :members: __init__, check_subset

.. autoclass:: pyomo.contrib.doe.scenario.Scenario_generator
    :special-members: __init__

.. autoclass:: pyomo.contrib.doe.result.FisherResults
    :special-members: __init__, calculate_FIM

.. autoclass:: pyomo.contrib.doe.result.GridSearchResult
    :special-members: __init__



Pyomo.DoE Usage Example
------------------------

We illustrate the Pyomo.DoE interface with a reaction kinetics example when feed A is converts to species B and C (Wang and Dowling, 2022).
Assuming an Arrhenius temperature dependence for the reaction rates :math:`k_1, k_2`, first-order reaction mechanisms, and only species A is fed to the reactor gives the following DAE model:


.. math::
    \begin{equation}
    \begin{aligned}
        k_1 & = A_1 e^{-\frac{E_1}{RT}} \\
        k_2 & = A_2 e^{-\frac{E_2}{RT}} \\
        \frac{d{C_A}}{dt} & = -k_1{C_A}  \\
        \frac{d{C_B}}{dt} &  = k_1{C_A} - k_2{C_B}  \\
        C_{A0}& = C_A + C_B + C_C \\
        C_B(t_0) & = 0 \\
        C_C(t_0) & = 0 \\
    \end{aligned}
    \end{equation}



Here :math:`C_A(t), C_B(t), C_C(t)` are the time-varying concentrations of the species A, B, C, respectively. The reaction rates
:math:`k_1, k_2` depend on the activation energies :math:`E_1, E_2` and pre-exponential factors :math:`A_1, A_2`.
The goal of MBDoE is to optimize the experiment design variables :math:`\boldsymbol{\varphi} = (C_{A0}, T(t))`, where :math:`C_{A0},T(t)` are the initial concentration of species A and the time-varying reactor temperature, to maximize the precision of unknown model parameters :math:`\boldsymbol{\theta} = (A_1, E_1, A_2, E_2)` by measuring :math:`\mathbf{y}(t)=(C_A(t), C_B(t), C_C(t))`.
The observation errors are assumed to be independent in both time and across measurements with a constant standard deviation of 1 M for each species. Thus the measurement covariance matrix is the identity matrix.


Step 0: Import Pyomo and the Pyomo.DoE module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doctest::

    >>> # === Required import ===
    >>> import pyomo.environ as pyo
    >>> from pyomo.dae import ContinuousSet, DerivativeVar
    >>> from pyomo.contrib.doe import Measurements, DesignOfExperiments
    >>> import numpy as np

Step 1: Define the Pyomo process model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The process model for the reaction kinetics problem is shown below.

.. doctest::

    >>> def create_model(scena, CA_init=5, T_initial=300,args=None):
    ...     # === Create model ==
    ...     m = pyo.ConcreteModel()
    ...     m.R = 8.31446261815324  # J/K/mol
    ...     # === Define set ===
    ...     m.t0 = pyo.Set(initialize=[0])
    ...     m.t = ContinuousSet(bounds=(0, 1))
    ...     m.t_con = pyo.Set(initialize=[0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1])
    ...     m.scena = pyo.Set(initialize=scena['scena-name'])
    ...     m.y_set = pyo.Set(initialize=['CA', 'CB', 'CC'])
    ...     # === Define variables ===
    ...     m.CA0 = pyo.Var(m.t0, initialize = CA_init, bounds=(1.0,5.0), within=pyo.NonNegativeReals) # mol/L
    ...     m.T = pyo.Var(m.t, initialize =T_initial, bounds=(300, 700), within=pyo.NonNegativeReals)
    ...     m.C = pyo.Var(m.scena, m.y_set, m.t, initialize=3, within=pyo.NonNegativeReals)
    ...     m.dCdt = DerivativeVar(m.C, wrt=m.t)
    ...     m.kp1 = pyo.Var(m.scena, m.t, initialize=3)
    ...     m.kp2 = pyo.Var(m.scena, m.t, initialize=1)
    ...     # === Define Param ===
    ...     m.A1 = pyo.Param(m.scena, initialize=scena['A1'],mutable=True)
    ...     m.A2 = pyo.Param(m.scena, initialize=scena['A2'],mutable=True)
    ...     m.E1 = pyo.Param(m.scena, initialize=scena['E1'],mutable=True)
    ...     m.E2 = pyo.Param(m.scena, initialize=scena['E2'],mutable=True)
    ...     # === Constraints ===
    ...     def T_control(m,t):
    ...         if t in m.t_con:
    ...             return pyo.Constraint.Skip
    ...         else:
    ...             j = -1
    ...             for t_con in m.t_con:
    ...                 if t>t_con:
    ...                     j+=1
    ...             neighbour_t = t_control[j]
    ...         return m.T[t] == m.T[neighbour_t]
    ...     def cal_kp1(m,z,t):
    ...         return m.kp1[z,t] == m.A1[z]*pyo.exp(-m.E1[z]*1000/(m.R*m.T[t]))
    ...     def cal_kp2(m,z,t):
    ...         return m.kp2[z,t] == m.A2[z]*pyo.exp(-m.E2[z]*1000/(m.R*m.T[t]))
    ...     def dCdt_control(m,z,y,t):
    ...         if y=='CA':
    ...             return m.dCdt[z,y,t] == -m.kp1[z,t]*m.C[z,'CA',t]
    ...         elif y=='CB':
    ...             return m.dCdt[z,y,t] == m.kp1[z,t]*m.C[z,'CA',t] - m.kp2[z,t]*m.C[z,'CB',t]
    ...         elif y=='CC':
    ...             return pyo.Constraint.Skip
    ...     def alge(m,z,t):
    ...         return m.C[z,'CA',t] + m.C[z,'CB',t] + m.C[z,'CC', t] == m.CA0[0]
    ...     m.T_rule = pyo.Constraint(m.t, rule=T_control)
    ...     m.k1_pert_rule = pyo.Constraint(m.scena, m.t, rule=cal_kp1)
    ...     m.k2_pert_rule = pyo.Constraint(m.scena, m.t, rule=cal_kp2)
    ...     m.dCdt_rule = pyo.Constraint(m.scena,m.y_set, m.t, rule=dCdt_control)
    ...     m.alge_rule = pyo.Constraint(m.scena, m.t, rule=alge)
    ...     for z in m.scena:
    ...         m.C[z,'CB',0.0].fix(0.0)
    ...         m.C[z,'CC',0.0].fix(0.0)
    ...     return m

Next we define a function to discretize the model.

.. doctest::

    >>> # === Discretization ===
    >>> def disc(m, NFE=32):
    ...     discretizer = pyo.TransformationFactory('dae.collocation')
    ...     discretizer.apply_to(m, nfe=NFE, ncp=3, wrt=m.t)
    ...     return m

.. note::
    The first argument of the ``create_model`` function should be ``scena``.

.. note::
    To use ``direct_kaug`` mode, the model parameters (  :math:`A_1, A_2, E_1, E_2`) definitations should be changes from ``Param`` to ``Var`` objects.

Step 2: Define the inputs for Pyomo.DoE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doctest::

    >>> # === Design variables, time points
    >>> t_control = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]    # Control time set [h]
    >>> dv_pass = {'CA0': [0],'T': t_control}  # design variable and its control time set

    >>> # === Measurement object ===
    >>> t_measure = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]   # Measurement time points [h]
    >>> measure_pass = {'C':{'CA': t_measure, 'CB': t_measure, 'CC': t_measure}}
    >>> measure_variance = {'C': {'CA': 1, 'CB': 1, 'CC': 1}} # provide measurement uncertainty
    >>> measure_class =  Measurements(measure_pass, variance=measure_variance)  # Use Pyomo.DoE.Measurements to achieve a measurement object

    >>> # === Parameter dictionary ===
    >>> parameter_dict = {'A1': 84.79, 'A2': 371.72, 'E1': 7.78, 'E2': 15.05}

    >>> # === Define prior information ==
    >>> prior_none = np.zeros((4,4))


Step 3: Compute the FIM of a square MBDoE problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This method computes an MBDoE optimization problem with no degree of freedom.

This method can be accomplished by two modes, ``direct_kaug`` and ``sequential_finite``.
``direct_kaug`` mode requires the installation of the solver `k_aug <https://github.com/dthierry/k_aug>`_ which is availabe through the `IDAES-PSE extensions <https://idaes-pse.readthedocs.io/en/stable/reference_guides/commands/get_extensions.html>`_.

.. doctest::

    >>> # === Decide mode ===
    >>> sensi_opt = 'sequential_finite'
    >>> # === Specify an experiment ===
    >>> exp1 = {'CA0': {0: 5}, 'T': {0: 570, 0.125:300,  0.25:300,  0.375:300,  0.5:300,  0.625:300,  0.75:300,  0.875:300, 1:300}}
    >>> # === Create the DoE object ===
    >>> doe_object = DesignOfExperiments(parameter_dict, dv_pass, measure_class, create_model,
    ...                            prior_FIM=prior_none, discretize_model=disc)
    >>> # === Use ``compute_FIM`` to compute one MBDoE square problem ===
    >>> result = doe_object.compute_FIM(exp1,mode=sensi_opt, FIM_store_name = 'dynamic.csv',
    ...                            store_output = 'store_output') # doctest: +SKIP
    >>> # === Use ``calculate_FIM`` method of the result object to evaluate the FIM ===
    >>> result.calculate_FIM(doe_object.design_values) # doctest: +SKIP
    >>> # === Print FIM and its trace, determinant, condition number and minimal eigen value ===
    >>> result.FIM  # doctest: +SKIP
    >>> result.trace # doctest: +SKIP
    >>> result.det # doctest: +SKIP
    >>> result.cond # doctest: +SKIP
    >>> result.min_eig  # doctest: +SKIP


Step 4: Exploratory analysis (Enumeration)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Exploratory analysis is suggested to enumerate the design space to check if the problem is identifiable, i.e., ensure that D-, E-optimality metrics are not small numbers near zero, and Modified E-optimality is not a big number.

Pyomo.DoE accomplishes the exploratory analysis with the ``run_grid_search`` function.
It allows users to define any number of design decisions. Heatmaps (or lines) are used the visualize the sensitivity of the DoE criteria to changes in two (or one) design variables with any other design variables held constant.
The function ``run_grid_search`` enumerates over the design space, each MBDoE problem accomplished by ``compute_FIM`` method. Therefore, ``run_grid_search`` supports only two modes: ``sequential_finite`` and ``direct_kaug``.


.. doctest::

    >>> # === Specify inputs===
    >>> design_ranges = [[1,2,3,4,5], [300,400,500,600,700]] # [CA0 [M], T [K]]
    >>> dv_apply_name = ['CA0','T']
    >>> dv_apply_time = [[0],t_control]
    >>> exp1 = {'CA0': {0: 5}, 'T': {0: 570, 0.125:300,  0.25:300,  0.375:300,  0.5:300,  0.625:300,  0.75:300,  0.875:300, 1:300}} # CA0 in [M], T in [K]
    >>> sensi_opt = 'sequential_finite'
    >>> prior_all = np.zeros((4,4))
    >>> prior_pass=np.asarray(prior_all)

    >>> # === Run enumeration ===
    >>> doe_object = DesignOfExperiments(parameter_dict, dv_pass, measure_class, create_model,
    ...                            prior_FIM=prior_none, discretize_model=disc) # doctest: +SKIP
    >>> all_fim = doe_object.run_grid_search(exp1, design_ranges, dv_apply_name, dv_apply_time, mode=sensi_opt) # doctest: +SKIP

    >>> # === Analyze results ===
    >>> test = all_fim.extract_criteria() # doctest: +SKIP
    >>> # === Draw 1D sensitivity curve===
    >>> fixed = {"'CA0'": 5.0} # fix a dimension
    >>> all_fim.figure_drawing(fixed, ['T'], 'Reactor case','T [K]','$C_{A0}$ [M]' ) # doctest: +SKIP
    >>> # === Draw 2D heatmap ===
    >>> fixed = {} # do not need to fix
    >>> all_fim.figure_drawing(fixed, ['CA0','T'], 'Reactor case','$C_{A0}$ [M]', 'T [K]' ) # doctest: +SKIP


Successful run of the above code shows the following figure:

.. figure:: grid-1.png
   :scale: 35 %

This heatmap shows the sensitivity of the DoE criteria, i.e., measures of the experimental information content, across the two-dimensional experiment design space. Horizontal and vertical axes are two design variables, while the color of each grid shows the experimental information content. For example, A-optimality shows that the most informative region is around :math:`C_{A0}=5.0` M, :math:`T=300.0` K, while the least informative region is around :math:`C_{A0}=1.0` M, :math:`T=700.0` K.

Step 5: Gradient-based optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pyomo.DoE formulates a two-stage stochastic_program to compute A- and D-optimality designs.

This function solves twice to ensure reliable intialization: first, Pyomo.DoE solves a square problem. Next, Pyomo.DoE unfixes the design variables (adds degrees of freedoms) and solves again.

.. doctest::

    >>> # === Specify a starting point ===
    >>> exp1 = {'CA0': {0: 5}, 'T': {0: 300, 0.125:300,  0.25:300,  0.375:300,  0.5:300,  0.625:300,  0.75:300,  0.875:300, 1:300}}
    >>> # === Define DoE object ===
    >>> doe_object = DesignOfExperiments(parameter_dict, dv_pass, measure_class, createmod,
    ...                            prior_FIM=prior_pass, discretize_model=disc) # doctest: +SKIP
    >>> # === Optimize ===
    >>> square_result, optimize_result= doe_object.stochastic_program(exp1,
    ...                                                         if_optimize=True,
    ...                                                          if_Cholesky=True,
    ...                                                         scale_nominal_param_value=True,
    ...                                                         objective_option='det',
    ...                                                         L_initial=None) # doctest: +SKIP


