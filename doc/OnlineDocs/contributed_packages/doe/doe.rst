Pyomo.DoE
=========

**Pyomo.DoE** (Pyomo Design of Experiments) is a Python library for model-based design of experiments using science-based models.

Pyomo.DoE was developed by **Jialu Wang** and **Alexander W. Dowling** at the University of Notre Dame as part of the `Carbon Capture Simulation for Industry Impact (CCSI2) <https://www.acceleratecarboncapture.org/>`_.
project, funded through the U.S. Department Of Energy Office of Fossil Energy.

If you use Pyomo.DoE, please cite:

[Wang and Dowling, 2022] Wang, Jialu, and Alexander W. Dowling. 
"Pyomo. DOE: An open‐source package for model‐based design of experiments in Python." 
AIChE Journal 68.12 (2022): e17813. `https://doi.org/10.1002/aic.17813`

Methodology Overview
---------------------

Model-based Design of Experiments (MBDoE) is a technique to maximize the information gain of experiments by directly using science-based models with physically meaningful parameters. It is one key component in the model calibration and uncertainty quantification workflow shown below:

.. figure:: flowchart.png
   :scale: 25 %

   The exploratory analysis, parameter estimation, uncertainty analysis, and MBDoE are combined into an iterative framework to select, refine, and calibrate science-based mathematical models with quantified uncertainty. Currently, Pyomo.DoE focused on increasing parameter precision.

Pyomo.DoE provides the exploratory analysis and MBDoE capabilities to the Pyomo ecosystem. The user provides one Pyomo model, a set of parameter nominal values,
the allowable design spaces for design variables, and the assumed observation error model.
During exploratory analysis, Pyomo.DoE checks if the model parameters can be inferred from the postulated measurements or preliminary data.
MBDoE then recommends optimized experimental conditions for collecting more data.
Parameter estimation packages such as `Parmest <https://pyomo.readthedocs.io/en/stable/contributed_packages/parmest/index.html>`_ can perform parameter estimation using the available data to infer values for parameters,
and facilitate an uncertainty analysis to approximate the parameter covariance matrix.
If the parameter uncertainties are sufficiently small, the workflow terminates and returns the final model with quantified parametric uncertainty.
If not, MBDoE recommends optimized experimental conditions to generate new data.

Below is an overview of the type of optimization models Pyomo.DoE can accomodate:

* Pyomo.DoE is suitable for optimization models of **continuous** variables
* Pyomo.DoE can handle **equality constraints** defining state variables
* Pyomo.DoE supports (Partial) Differential-Algebraic Equations (PDAE) models via Pyomo.DAE
* Pyomo.DoE also supports models with only algebraic constraints

The general form of a DAE problem that can be passed into Pyomo.DoE is shown below:

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
    * Process models provided to Pyomo.DoE should define an extra scenario index for all state variables and all parameters, as the first index before any other index.
    * Process models must include an index for time, named ``t``. For steady-state models, t should be [0].
    * Measurements can have an extra index besides time.
    * Parameters and design variables should be defined as Pyomo ``var`` components on the model to use ``direct_kaug`` mode, and can be defined as Pyomo ``Param`` object if not using ``direct_kaug``.
    * Create model function should take scenarios as the first argument of this function.
    * Design variables are defined with and only with a time index.

Based on the above notation, the form of the MBDoE problem addressed in Pyomo.DoE is shown below:

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

Pyomo.DoE provides four design criteria  :math:`\Psi` to measure the size of FIM:

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

In order to solve problems of the above, Pyomo.DoE implements the 2-stage stochastic program. Please see Wang and Dowling (2022) for details.

Pyomo.DoE Required Inputs
--------------------------------
The required inputs to the Pyomo.DoE solver are the following:

* A function that creates the process model
* Dictionary of parameters and their nominal value
* A measurement object
* A design variables object
* A Numpy ``array`` containing the Prior FIM
* Optimization solver

Below is a list of arguments that Pyomo.DoE expects the user to provide.

param_init : ``dictionary``
    A ``dictionary`` of parameter names and values. If they are an indexed variable, put the variable name and index in a nested ``Dictionary``.

design_variable_timepoints : ``object``
    A ``object`` of design variables, provided by the design class.
    If this design var is independent of time (constant), set the time to [0]

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
        #.  Fix the experiment design decisions and solve a square (i.e., zero degrees of freedom) instance of the two-stage DOE problem. This step is for initialization.
        #.  Unfix the experiment design decisions and solve the two-stage DOE problem.

.. autoclass:: pyomo.contrib.doe.measurements.Measurements
    :members: __init__, add_elements

.. autoclass:: pyomo.contrib.doe.measurements.DesignVariabless
    :members: __init__, add_elements

.. autoclass:: pyomo.contrib.doe.scenario.ScenarioGenerator
    :special-members: __init__

.. autoclass:: pyomo.contrib.doe.result.FisherResults
    :special-members: __init__, calculate_FIM

.. autoclass:: pyomo.contrib.doe.result.GridSearchResult
    :special-members: __init__



Pyomo.DoE Usage Example
------------------------

We illustrate the use of Pyomo.DoE using a reaction kinetics example (Wang and Dowling, 2022).
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


Step 0: Import Pyomo and the Pyomo.DoE module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doctest::

    >>> # === Required import ===
    >>> import pyomo.environ as pyo
    >>> from pyomo.dae import ContinuousSet, DerivativeVar
    >>> from pyomo.contrib.doe import DesignOfExperiments, Measurements, DesignVariables
    >>> from pyomo.contrib.doe import objective_lib, finite_difference_lib, calculation_mode
    >>> import numpy as np

Step 1: Define the Pyomo process model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The process model for the reaction kinetics problem is shown below.

.. doctest::

    >>> def create_model(mod=None, model_option="block"):
    ...     # === model information === 
    ...     theta = {'A1': 84.79, 'A2': 371.72, 'E1': 7.78, 'E2': 15.05} # parameters
    ...     t_control =  [0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1] # control timepoints
    ...     # === model options ===
    ...     if model_option == "parmest":
    ...         mod = pyo.ConcreteModel()
    ...         return_m = True # the created model will be returned
    ...     elif model_option in ["block", "global"]: # a model needs to be passed
    ...         if not mod:
    ...             raise ValueError("If model option is global or block, a created model needs to be provided.")
    ...         return_m = False # no need to return the model 
    ...     else:
    ...         raise ValueError("model_option needs to be defined as global, block, or parmest.")
    ...     
    ...     mod.t0 = pyo.Set(initialize=[0])
    ...     mod.t_con = pyo.Set(initialize=t_control)
    ...     mod.CA0 = pyo.Var(mod.t0, initialize = 1, bounds=(1.0,5.0), within=pyo.NonNegativeReals) # mol/L
    ...     if model_option=="global":
    ...         mod.T = pyo.Var(mod.t_con, initialize = 300, bounds=(300, 700), within=pyo.NonNegativeReals)
    ...     else:
    ...         para_list = ['A1', 'A2', 'E1', 'E2']
    ...         mod.t = ContinuousSet(bounds=(0, 1))
    ...         mod.T = pyo.Var(mod.t, initialize =300, bounds=(300, 700), within=pyo.NonNegativeReals)
    ...         mod.R = 8.31446261815324 # J / K / mole
    ...         # Define parameters as Param
    ...         mod.A1 = pyo.Var(initialize = theta['A1'])
    ...         mod.A2 = pyo.Var(initialize = theta['A2'])
    ...         mod.E1 = pyo.Var(initialize = theta['E1'])
    ...         mod.E2 = pyo.Var(initialize = theta['E2'])
    ...         # Concentration variables under perturbation
    ...         mod.C_set = pyo.Set(initialize=['CA','CB','CC'])
    ...         mod.C = pyo.Var(mod.C_set, mod.t, initialize=C_init, within=pyo.NonNegativeReals)
    ...         # time derivative of C
    ...         mod.dCdt = DerivativeVar(mod.C, wrt=mod.t)  
    ...         # kinetic parameters
    ...         def kp1_init(m,t):
    ...             return m.A1 * pyo.exp(-m.E1*1000/(m.R*m.T[t]))
    ...         def kp2_init(m,t):
    ...             return m.A2 * pyo.exp(-m.E2*1000/(m.R*m.T[t]))
    ...         mod.kp1 = pyo.Var(mod.t, initialize=kp1_init)
    ...         mod.kp2 = pyo.Var(mod.t, initialize=kp2_init)
    ...         # === Constraints ===
    ...         def T_control(m,t):
    ...             if t in m.t_con:
    ...                 return pyo.Constraint.Skip
    ...             else:
    ...                 j = -1
    ...                 for t_con in m.t_con:
    ...                     if t>t_con:
    ...                         j+=1
    ...                 neighbour_t = t_control[j]
    ...                 return m.T[t] == m.T[neighbour_t]
    ...         def cal_kp1(m,t):
    ...             return m.kp1[t] == m.A1*pyo.exp(-m.E1*1000/(m.R*m.T[t])) 
    ...         def cal_kp2(m,t):
    ...             return m.kp2[t] == m.A2*pyo.exp(-m.E2*1000/(m.R*m.T[t])) 
    ...         def dCdt_control(m,y,t):
    ...             if y=='CA':
    ...                 return m.dCdt[y,t] == -m.kp1[t]*m.C['CA',t]
    ...             elif y=='CB':
    ...                 return m.dCdt[y,t] == m.kp1[t]*m.C['CA',t] - m.kp2[t]*m.C['CB',t]
    ...             elif y=='CC':
    ...                 return pyo.Constraint.Skip
    ...         def alge(m,t):
    ...             return m.C['CA',t] + m.C['CB',t] + m.C['CC', t] == m.CA0[0]
    ...         mod.T_rule = pyo.Constraint(mod.t, rule=T_control)
    ...         mod.k1_pert_rule = pyo.Constraint(mod.t, rule=cal_kp1)
    ...         mod.k2_pert_rule = pyo.Constraint(mod.t, rule=cal_kp2)
    ...         mod.dCdt_rule = pyo.Constraint(mod.C_set, mod.t, rule=dCdt_control)
    ...         mod.alge_rule = pyo.Constraint(mod.t, rule=alge)
    ...         mod.C['CB',0.0].fix(0.0)
    ...         mod.C['CC',0.0].fix(0.0)
    ...     if return_m:
    ...         return m
.. doctest::

    >>> # === Discretization ===
    >>> def disc_for_measure(m, NFE=32, block=True):
    ...     discretizer = pyo.TransformationFactory('dae.collocation')
    ...     if block:
    ...         for s in range(len(m.block)):
    ...             discretizer.apply_to(m.block[s], nfe=NFE, ncp=3, wrt=m.block[s].t)
    ...     else: 
    ...         discretizer.apply_to(m, nfe=NFE, ncp=3, wrt=m.t)
    ...     return m

.. note::
    The model requires at least two options: "block" and "global". Both options requires the pass of a created empty Pyomo model. 
    With "global" option, only design variables and their time sets need to be defined; 
    With "block" option, a full model needs to be defined. 


Step 2: Define the inputs for Pyomo.DoE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doctest::

    >>> t_control = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]    # Control time set [h]

    >>> # === Measurement object ===
    >>> total_name = ["C"]   # Measurement names 
    >>> extra_index = [[["CA", "CB", "CC"]]] # extra indexes
    >>> time_index = [t_control]  # time list 
    >>> measure_class = Measurements()  # Use Pyomo.DoE.Measurements to achieve a measurement object
    >>> measure_class.add_elements(total_name, extra_index=extra_index, time_index = time_index)

    >>> # === Parameter dictionary ===
    >>> parameter_dict = {'A1': 84.79, 'A2': 371.72, 'E1': 7.78, 'E2': 15.05}

    >>> # === Design variables object === 
    >>> total_name = ["CA0", "T"] # names
    >>> dtime_index = [[0], t_control] # time indexes 
    >>> exp1 = [5, 570, 300, 300, 300, 300, 300, 300, 300, 300]
    >>> upper_bound = [5, 700, 700, 700, 700, 700, 700, 700, 700, 700]
    >>> lower_bound = [1, 300, 300, 300, 300, 300, 300, 300, 300, 300]

    >>> design_gen = DesignVariables() # create object
    >>> design_gen.add_elements(total_name, time_index = dtime_index, values=exp1, 
    ... upper_bound=upper_bound, lower_bound=lower_bound)

    >>> # === Define prior information ==
    >>> prior_pass = np.zeros((4,4))


Step 3: Compute the FIM of a square MBDoE problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This method computes an MBDoE optimization problem with no degree of freedom.

This method can be accomplished by two modes, ``direct_kaug`` and ``sequential_finite``.
``direct_kaug`` mode requires the installation of the solver `k_aug <https://github.com/dthierry/k_aug>`_.

.. doctest::

    >>> # === Decide mode ===
    >>> sensi_opt = calculation_mode.sequential_finite
    >>> # === Specify an experiment ===
    >>> exp1 = [5, 570, 300, 300, 300, 300, 300, 300, 300, 300]
    >>> design_gen.update_values(exp1) # update values for design object
    >>> # === Create the DOE object ===
    >>> doe_object = DesignOfExperiments(parameter_dict, design_gen,
    ...                          measure_class, create_model,
    ...                         prior_FIM=prior_pass, discretize_model=disc_for_measure)
    >>> # === Use ``compute_FIM`` to compute one MBDoE square problem ===
    >>> result = doe_object.compute_FIM(mode=sensi_opt,  scale_nominal_param_value=True,
    ...                            formula = finite_difference_lib.central) # doctest: +SKIP
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

Exploratory analysis is suggested to enumerate the design space to check if the problem is identifiable, 
i.e., ensure that D-, E-optimality metrics are not small numbers near zero, and Modified E-optimality is not a big number.

Pyomo.DoE accomplishes the exploratory analysis with the ``run_grid_search`` function.
It allows users to define any number of design decisions. Heatmaps can be drawn by two design variables, fixing other design variables.
1D curve can be drawn by one design variable, fixing all other variables.
The function ``run_grid_search`` enumerates over the design space, each MBDoE problem accomplished by ``compute_FIM`` method.
Therefore, ``run_grid_search`` supports only two modes: ``sequential_finite`` and ``direct_kaug``.


.. doctest::

    >>> # === Specify inputs===
    >>> design_ranges = [list(np.linspace(1,5,3)), list(np.linspace(300,700,3))] # CA0, T
    >>> dv_apply_name = ['CA0[0]',['T[0]','T[0.125]','T[0.25]','T[0.375]','T[0.5]','T[0.625]','T[0.75]','T[0.875]','T[1]']]
    >>> sensi_opt = calculation_mode.direct_kaug
    >>> prior_pass = np.zeros((4,4))

    >>> # === Run enumeration ===
    >>> doe_object = DesignOfExperiments(parameter_dict, design_gen,measure_class, create_model,
    ...                            prior_FIM=prior_pass, discretize_model=disc_for_measure) # doctest: +SKIP
    >>> all_fim = doe_object.run_grid_search(design_ranges, dv_apply_name, mode=sensi_opt) # doctest: +SKIP

    >>> # === Analyze results ===
    >>> all_fim.extract_criteria() # doctest: +SKIP
    >>> # === Draw 1D sensitivity curve===
    >>> fixed = {"'CA0[0]'": 5.0} # fix a dimension
    >>> all_fim.figure_drawing(fixed, ['T[0]'], 'Reactor case','T [K]','$C_{A0}$ [M]' ) # doctest: +SKIP
    >>> # === Draw 2D heatmap ===
    >>> fixed = {} # do not need to fix
    >>> all_fim.figure_drawing(fixed, ['CA0[0]','T[0]'], 'Reactor case','$C_{A0}$ [M]', 'T [K]' ) # doctest: +SKIP


Successful run of the above code shows the following figure:

.. figure:: grid-1.png
   :scale: 35 %

A heatmap shows the change of the objective function, a.k.a. the experimental information content, in the design region. Horizontal and vertical axes are two design variables, while the color of each grid shows the experimental information content. Taking the Fig. Reactor case - A optimality as example, A-optimality shows that the most informative region is around $C_{A0}=5.0$ M, $T=300.0$ K, while the least informative region is around $C_{A0}=1.0$ M, $T=700.0$ K.

Step 5: Gradient-based optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pyomo.DoE accomplishes gradient-based optimization with the ``stochastic_program`` function for A- and D-optimality design.

This function solves twice: It solves the square version of the MBDoE problem first, and then unfixes the design variables as degree of freedoms and solves again. In this way the optimization problem can be well initialized.

.. doctest::

    >>> # === Specify a starting point ===
    >>> exp1 = [5, 570, 300, 300, 300, 300, 300, 300, 300, 300]
    >>> design_gen.update_values(exp1)
    >>> # === Define DOE object ===
    >>> doe_object = DesignOfExperiments(parameter_dict, design_gen,measure_class, create_model,
    ...                            prior_FIM=prior, discretize_model=disc_for_measure) # doctest: +SKIP
    >>> # === Optimize ===
    >>> square_result, optimize_result= doe_object2.stochastic_program(if_optimize=True, if_Cholesky=True,
    ...                                                         scale_nominal_param_value=True, objective_option=objective_lib.det, 
    ...                                                          L_initial=np.linalg.cholesky(prior)) # doctest: +SKIP


