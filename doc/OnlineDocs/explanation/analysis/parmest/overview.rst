Overview
========

The Python package called parmest facilitates model-based parameter
estimation along with characterization of uncertainty associated with
the estimates. For example, parmest can provide confidence regions
around the parameter estimates.  Additionally, parameter vectors, each
with an attached probability estimate, can be used to build scenarios
for design optimization.

Functionality in parmest includes:

* Model-based parameter estimation using experimental data
* Covariance matrix estimation
* Bootstrap resampling for parameter estimation
* Confidence regions based on single or multi-variate distributions
* Likelihood ratio
* Leave-N-out cross validation
* Parallel processing

Background
----------

The goal of parameter estimation is to estimate values for 
a vector, :math:`\boldsymbol{\theta}`, to use in the functional form

.. math::
      
   \boldsymbol{y}_i = \boldsymbol{f}\left(\boldsymbol{x}_{i}, \boldsymbol{\theta}\right) +
    \boldsymbol{\varepsilon}_i \quad \forall \; i \in \{1, \ldots, n\}

where :math:`\boldsymbol{y}_{i} \in \mathbb{R}^m` are observations of the measured or output variables,
:math:`\boldsymbol{f}` is the model function, :math:`\boldsymbol{x}_{i} \in \mathbb{R}^{q}` are the decision
or input variables, :math:`\boldsymbol{\theta} \in \mathbb{R}^p` are the model parameters,
:math:`\boldsymbol{\varepsilon}_{i} \in \mathbb{R}^m` are measurement errors, and :math:`n` is the number of
experiments.

The following least squares objective can be used to estimate parameter
values assuming Gaussian independent and identically distributed measurement
errors:

.. math::

   \min_{\boldsymbol{\theta}} \, g(\boldsymbol{x}, \boldsymbol{y};\boldsymbol{\theta}) \;\;

where :math:`g(\boldsymbol{x}, \boldsymbol{y};\boldsymbol{\theta})` can be:

1. Sum of squared errors

    .. math::

       g(\boldsymbol{x}, \boldsymbol{y};\boldsymbol{\theta}) =
        \sum_{i = 1}^{n} \left(\boldsymbol{y}_{i} - \boldsymbol{f}(\boldsymbol{x}_{i};\boldsymbol{\theta})
        \right)^\text{T} \left(\boldsymbol{y}_{i} - \boldsymbol{f}(\boldsymbol{x}_{i};\boldsymbol{\theta})\right)

2. Weighted sum of squared errors

    .. math::

       g(\boldsymbol{x}, \boldsymbol{y};\boldsymbol{\theta}) =
        \frac{1}{2} \sum_{i = 1}^{n} \left(\boldsymbol{y}_{i} - \boldsymbol{f}(\boldsymbol{x}_{i};\boldsymbol{\theta})
        \right)^\text{T} \boldsymbol{\Sigma}_{\boldsymbol{y}}^{-1} \left(\boldsymbol{y}_{i} -
        \boldsymbol{f}(\boldsymbol{x}_{i};\boldsymbol{\theta})\right)

where :math:`\boldsymbol{\Sigma}_{\boldsymbol{y}}` is the measurement error covariance matrix containing the
standard deviation of the measurement errors of :math:`\boldsymbol{y}`. Custom objectives can also be defined
for parameter estimation.

In the applications of interest to us, the function :math:`g(\cdot)` is
usually defined as an optimization problem with a large number of
(perhaps constrained) optimization variables, a subset of which are
fixed at values :math:`\boldsymbol{x}` when the optimization is performed.
In other applications, the values of :math:`\boldsymbol{\theta}` are fixed
parameter values, but for the problem formulation above, the values of
:math:`\boldsymbol{\theta}` are the primary optimization variables. Note that in
general, the function :math:`g(\cdot)` will have a large set of
parameters that are not included in :math:`\boldsymbol{\theta}`.
