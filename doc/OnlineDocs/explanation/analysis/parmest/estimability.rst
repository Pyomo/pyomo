.. _estimabilitysection:

Estimability Analysis
=====================

After estimating the model parameters with their associated uncertainty, as demonstrated in the
:ref:`driversection` and :ref:`covariancesection` Section, estimability analysis is required to identify
parameters that cannot be reliably estimated from the available data due to limitations in the mathematical
model structure. If such parameters are identified, the model may need to be reformulated, replaced with an
alternative structure, or augmented with additional prior information. In parmest, estimability analysis can
be performed using eigen-decomposition of the parameter covariance matrix, profile likelihood methods, or
multi-start initialization routines.

Eigen-decomposition
-------------------

The estimability of model parameters can be analyzed through eigen-decomposition of the covariance matrix
obtained from parameter estimation. This covariance matrix quantifies parameter uncertainty and captures both
parameter variances and correlations. Eigen-decomposition of this matrix identifies principal directions
in parameter space along which uncertainty is largest or smallest. These directions provide insight into
parameter identifiability and reveal combinations of parameters that are either structurally identifiable or
non-identifiable based on the underlying model formulation.

.. note::

   Detailed descriptions and example code for this method will be added in a future update.

Profile Likelihood
------------------

Profile likelihood analysis evaluates parameter estimability by systematically varying one parameter while
re-optimizing the remaining parameters to maintain consistency with the model and observed data. This approach
is closely related to likelihood ratio–based uncertainty quantification and provides a robust characterization
of practical identifiability through the shape of the likelihood surface. In addition, it can reveal structural
non-identifiability when flat or unbounded profiles indicate parameter combinations that are not uniquely
determined by the model formulation, particularly in nonlinear systems.

.. note::

   Detailed descriptions and example code for this method will be added in a future update.

Multi-start Initialization
--------------------------

Multi-start initialization assesses parameter estimability by exploring a range of initial guesses. Because
parameter estimation problems are often nonlinear and may exhibit multiple local minima, different initializations
can lead to different parameter estimates. By solving the estimation problem from multiple starting points, one can
evaluate the robustness of the solution and identify potential issues related to non-convexity or non-identifiability.
Consistent convergence to a unique solution across initializations suggests that the parameters are structurally
identifiable within the model formulation, whereas sensitivity to initialization indicates potential estimability
issues.

.. note::

   Detailed descriptions and example code for this method will be added in a future update.