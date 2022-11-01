Pyomo.DOE
=========

**Pyomo.DOE** (Pyomo Design of Experiments) is a Python library for model-based design of experiments using science-based models.

Pyomo.DOE was developed by **Jialu Wang** and **Alexander W. Dowling** at the University of Notre Dame as part of the `Carbon Capture Simulation for Industry Impact (CCSI2) <https://www.acceleratecarboncapture.org/>`_.
project, funded through the U.S. Department Of Energy Office of Fossil Energy.

If you use Pyomo.DOE, please cite:

[Wang and Dowling, 2022] Jialu Wang, Alexander Dowling (2022), Pyomo.DOE: An Open-Source Package for Model-Based Design of Experiments in Python. (under review)

Methodology Overview
---------------------

Model-based Design of Experiments (MBDoE) is a technique to maximize the information gain of experiments by directly using science-based models with physically meaningful parameters. It is one key component in the model calibration and uncertainty quantification workflow shown below:

.. figure:: flowchart.png
   :scale: 25 %

   The exploratory analysis, parameter estimation, uncertainty analysis, and MBDoE are combined into an iterative framework to select, refine, and calibrate science-based mathematical models with quantified uncertainty. Currently, Pyomo.DOE focused on increasing parameter precision.

Pyomo.DOE provides the exploratory analysis and MBDoE capabilities to the Pyomo ecosystem. The user provides one Pyomo model, a set of parameter nominal values,
the allowable design spaces for design variables, and the assumed observation error model.
During exploratory analysis, Pyomo.DOE checks if the model parameters can be inferred from the postulated measurements or preliminary data.
MBDoE then recommends optimized experimental conditions for collecting more data.
Parameter estimation packages such as `Parmest <https://pyomo.readthedocs.io/en/stable/contributed_packages/parmest/index.html>`_ can perform parameter estimation using the available data to infer values for parameters,
and facilitate an uncertainty analysis to approximate the parameter covariance matrix.
If the parameter uncertainties are sufficiently small, the workflow terminates and returns the final model with quantified parametric uncertainty.
If not, MBDoE recommends optimized experimental conditions to generate new data.

Below is an overview of the type of optimization models Pyomo.DOE can accomodate:

* Pyomo.DOE is suitable for optimization models of **continuous** variables
* Pyomo.DOE can handle **equality constraints** defining state variables
* Pyomo.DOE supports (Partial) Differential-Algebraic Equations (PDAE) models via Pyomo.DAE
* Pyomo.DOE also supports models with only algebraic constraints

The general form of a DAE problem that can be passed into Pyomo.DOE is shown below:

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
    * Process models provided to Pyomo.DOE should define an extra scenario index for all state variables and all parameters, as the first index before any other index.
    * Process models must include an index for time, named ``t``. For steady-state models, t should be [0].
    * Measurements can have an extra index besides time.
    * Parameters and design variables should be defined as Pyomo ``var`` components on the model to use ``direct_kaug`` mode, and can be defined as Pyomo ``Param`` object if not using ``direct_kaug``.
    * Create model function should take scenarios as the first argument of this function.
    * Design variables are defined with and only with a time index.

Based on the above notation, the form of the MBDoE problem addressed in Pyomo.DOE is shown below:

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
*  :math:`\Psi` is the design criteria to measure FIM.
*  :math:`\mathbf{V}_{\boldsymbol{\theta}}(\boldsymbol{\hat{\theta}})^{-1}` is the FIM of previous experiments.

Pyomo.DOE provides four design criteria  :math:`\Psi` to measure the size of FIM:

.. list-table:: Pyomo.DOE design criteria
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

In order to solve problems of the above, Pyomo.DOE implements the 2-stage stochastic program. Please see Wang and Dowling (2022) for details.

Pyomo.DOE Required Inputs
--------------------------------
The required inputs to the Pyomo.DOE solver are the following:

* A function that creates the process model
* Dictionary of parameters and their nominal value
* Dictionary of measurements and their measurement time points
* Dictionary of design variables and their control time points
* A Numpy ``array`` containing the Prior FIM
* Local and global optimization solver

Below is a list of arguments that Pyomo.DOE expects the user to provide.

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

Pyomo.DOE Solver Interface
---------------------------

.. figure:: uml.png
   :scale: 25 %


.. autoclass:: pyomo.contrib.doe.fim_doe.DesignOfExperiments
    :members: __init__, optimize_doe, compute_FIM, run_grid_search

.. Note::
    ``optimize_doe()`` includes the following steps:
        #.  Build two-stage stochastic programming optimization model where scenarios correspond to finite difference approximations for the Jacobian of the response variables with respect to calibrated model parameters
        #.  Fix the experiment design decisions and solve a square (i.e., zero degrees of freedom) instance of the two-stage DOE problem. This step is for initialization.
        #.  Unfix the experiment design decisions and solve the two-stage DOE problem.

.. autoclass:: pyomo.contrib.doe.fim_doe.Measurements
    :members: __init__, check_subset

.. autoclass:: pyomo.contrib.doe.fim_doe.Scenario_generator
    :special-members: __init__

.. autoclass:: pyomo.contrib.doe.fim_doe.FIM_result
    :special-members: __init__, calculate_FIM

.. autoclass:: pyomo.contrib.doe.fim_doe.Grid_Search_Result
    :special-members: __init__



Pyomo.DOE Usage Example
------------------------

We illustrate the use of Pyomo.DOE using a reaction kinetics example (Wang and Dowling, 2022).
The Arrhenius equations model the temperature dependence of the reaction rate coefficient  :math:`k_1, k_2`. Assuming a first-order reaction mechanism gives the reaction rate model. Further, we assume only species A is fed to the reactor.


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



:math:`C_A(t), C_B(t), C_C(t)` are the time-varying concentrations of the species A, B, C, respectively.
:math:`k_1, k_2` are the rates for the two chemical reactions using an Arrhenius equation with activation energies :math:`E_1, E_2` and pre-exponential factors :math:`A_1, A_2`.
The goal of MBDoE is to optimize the experiment design variables :math:`\boldsymbol{\varphi} = (C_{A0}, T(t))`, where :math:`C_{A0},T(t)` are the initial concentration of species A and the time-varying reactor temperature, to maximize the precision of unknown model parameters :math:`\boldsymbol{\theta} = (A_1, E_1, A_2, E_2)` by measuring :math:`\mathbf{y}(t)=(C_A(t), C_B(t), C_C(t))`.
The observation errors are assumed to be independent both in time and across measurements with a constant standard deviation of 1 M for each species.


Step 0: Import Pyomo and the Pyomo.DOE module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doctest::

    >>> # === Required import ===
    >>> import pyomo.environ as pyo
    >>> from pyomo.dae import ContinuousSet, DerivativeVar
    >>> import pyomo.contrib.doe.fim_doe as doe
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
.. doctest::

    >>> # === Discretization ===
    >>> def disc(m, NFE=32):
    ...     discretizer = pyo.TransformationFactory('dae.collocation')
    ...     discretizer.apply_to(m, nfe=NFE, ncp=3, wrt=m.t)
    ...     return m

.. note::
    The first argument of the ``create_model`` function should be ``scena``.

.. note::
    The model parameters (  :math:`A_1, A_2, E_1, E_2`) are defined as either ``Var`` or ``Param`` objects. This is suggested since it allows an easy transition when changing mode to ``direct_kaug``.

Step 2: Define the inputs for Pyomo.DOE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doctest::

    >>> # === Design variables, time points
    >>> t_control = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]    # Control time set [h]
    >>> dv_pass = {'CA0': [0],'T': t_control}  # design variable and its control time set

    >>> # === Measurement object ===
    >>> t_measure = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]   # Measurement time points [h]
    >>> measure_pass = {'C':{'CA': t_measure, 'CB': t_measure, 'CC': t_measure}}
    >>> measure_variance = {'C': {'CA': 1, 'CB': 1, 'CC': 1}} # provide measurement uncertainty
    >>> measure_class =  doe.Measurements(measure_pass, variance=measure_variance)  # Use Pyomo.DoE.Measurements to achieve a measurement object

    >>> # === Parameter dictionary ===
    >>> parameter_dict = {'A1': 84.79, 'A2': 371.72, 'E1': 7.78, 'E2': 15.05}

    >>> # === Define prior information ==
    >>> prior_none = np.zeros((4,4))


Step 3: Compute the FIM of a square MBDoE problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This method computes an MBDoE optimization problem with no degree of freedom.

This method can be accomplished by two modes, ``direct_kaug`` and ``sequential_finite``.
``direct_kaug`` mode requires the installation of the solver `k_aug <https://github.com/dthierry/k_aug>`_.

.. doctest::

    >>> # === Decide mode ===
    >>> sensi_opt = 'sequential_finite'
    >>> # === Specify an experiment ===
    >>> exp1 = {'CA0': {0: 5}, 'T': {0: 570, 0.125:300,  0.25:300,  0.375:300,  0.5:300,  0.625:300,  0.75:300,  0.875:300, 1:300}}
    >>> # === Create the DOE object ===
    >>> doe_object = doe.DesignOfExperiments(parameter_dict, dv_pass, measure_class, create_model,
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

Pyomo.DOE accomplishes the exploratory analysis with the ``run_grid_search`` function.
It allows users to define any number of design decisions. Heatmaps can be drawn by two design variables, fixing other design variables.
1D curve can be drawn by one design variable, fixing all other variables.
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
    >>> doe_object = doe.DesignOfExperiments(parameter_dict, dv_pass, measure_class, create_model,
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


Step 5: Gradient-based optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pyomo.DOE accomplishes gradient-based optimization with the ``optimize_doe`` function for A- and D-optimality design.

This function solves twice: It solves the square version of the MBDoE problem first, and then unfixes the design variables as degree of freedoms and solves again. In this way the optimization problem can be well initialized.

.. doctest::

    >>> # === Specify a starting point ===
    >>> exp1 = {'CA0': {0: 5}, 'T': {0: 300, 0.125:300,  0.25:300,  0.375:300,  0.5:300,  0.625:300,  0.75:300,  0.875:300, 1:300}}
    >>> # === Define DOE object ===
    >>> doe_object = doe.DesignOfExperiments(parameter_dict, dv_pass, measure_class, createmod,
    ...                            prior_FIM=prior_pass, discretize_model=disc) # doctest: +SKIP
    >>> # === Optimize ===
    >>> square_result, optimize_result= doe_object.optimize_doe(exp1,
    ...                                                         if_optimize=True,
    ...                                                          if_Cholesky=True,
    ...                                                         scale_nominal_param_value=True,
    ...                                                         objective_option='det',
    ...                                                         L_initial=None) # doctest: +SKIP
    >>> # === Analyze results===
    # TODO: single print statments with multiple lines.
    >>> print('This optimization is solved with status:', optimize_result.status) # doctest: +SKIP
    >>> print('The result FIM is:', optimize_result.FIM) # doctest: +SKIP
    >>> print('Four design criteria log10() value:') # doctest: +SKIP
    >>> print('A-optimality:', np.log10(optimize_result.trace)) # doctest: +SKIP
    >>> print('D-optimality:', np.log10(optimize_result.det)) # doctest: +SKIP
    >>> print('E-optimality:', np.log10(optimize_result.min_eig)) # doctest: +SKIP
    >>> print('Modified E-optimality:', np.log10(optimize_result.cond)) # doctest: +SKIP


