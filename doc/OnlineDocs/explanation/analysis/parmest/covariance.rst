.. _covariancesection:

Covariance Matrix Estimation
============================

The goal of parameter estimation (see :ref:`driversection` Section) is to estimate unknown model parameters
from experimental data. When the model parameters are estimated from the data, their accuracy is measured by
computing the covariance matrix. The diagonal of this covariance matrix contains the variance of the
estimated parameters which is used to calculate their uncertainty. Assuming Gaussian independent and identically
distributed measurement errors, the covariance matrix of the estimated parameters can be computed using the
following methods which have been implemented in parmest.

1. Reduced Hessian Method

    When the objective function is the sum of squared errors (SSE) for homogeneous data, defined as
    :math:`\text{SSE} = \sum_{i = 1}^{n} \left(\boldsymbol{y}_{i} - \boldsymbol{f}(\boldsymbol{x}_{i};
    \boldsymbol{\theta})\right)^\text{T} \left(\boldsymbol{y}_{i} - \boldsymbol{f}(\boldsymbol{x}_{i};
    \boldsymbol{\theta})\right)`, the covariance matrix is:

    .. math::
       \boldsymbol{V}_{\boldsymbol{\theta}} = 2 \sigma^2 \left(\frac{\partial^2 \text{SSE}}
        {\partial \boldsymbol{\theta}^2}\right)^{-1}_{\boldsymbol{\theta}
        = \hat{\boldsymbol{\theta}}}

    Similarly, when the objective function is the weighted SSE (WSSE) for heterogeneous data, defined as
    :math:`\text{WSSE} = \frac{1}{2} \sum_{i = 1}^{n} \left(\boldsymbol{y}_{i} -
    \boldsymbol{f}(\boldsymbol{x}_{i};\boldsymbol{\theta})\right)^\text{T} \boldsymbol{\Sigma}_{\boldsymbol{y}}^{-1}
    \left(\boldsymbol{y}_{i} - \boldsymbol{f}(\boldsymbol{x}_{i};\boldsymbol{\theta})\right)`,
    the covariance matrix is:

    .. math::
       \boldsymbol{V}_{\boldsymbol{\theta}} = \left(\frac{\partial^2 \text{WSSE}}
        {\partial \boldsymbol{\theta}^2}\right)^{-1}_{\boldsymbol{\theta}
        = \hat{\boldsymbol{\theta}}}

    Where :math:`\boldsymbol{V}_{\boldsymbol{\theta}}` is the covariance matrix of the estimated
    parameters :math:`\hat{\boldsymbol{\theta}} \in \mathbb{R}^p`, :math:`\boldsymbol{y}_{i} \in \mathbb{R}^m` are
    observations of the measured variables, :math:`\boldsymbol{f}` is the model function,
    :math:`\boldsymbol{x}_{i} \in \mathbb{R}^{q}` are the input variables, :math:`n` is the number of experiments,
    :math:`\boldsymbol{\Sigma}_{\boldsymbol{y}}` is the measurement error covariance matrix, and :math:`\sigma^2`
    is the variance of the measurement error. When the standard deviation of the measurement error is not supplied
    by the user, parmest approximates :math:`\sigma^2` as:
    :math:`\hat{\sigma}^2 = \frac{1}{n-p} \sum_{i=1}^{n} \boldsymbol{\varepsilon}_{i}(\boldsymbol{\theta})^{\text{T}}
    \boldsymbol{\varepsilon}_{i}(\boldsymbol{\theta})`, and :math:`\boldsymbol{\varepsilon}_{i} \in \mathbb{R}^m`
    are the residuals between the data and model for experiment :math:`i`.

    In parmest, this method computes the inverse of the Hessian by scaling the
    objective function (SSE or WSSE) with a constant probability factor, :math:`\frac{1}{n}`.

2. Finite Difference Method

    In this method, the covariance matrix, :math:`\boldsymbol{V}_{\boldsymbol{\theta}}`, is
    computed by differentiating the Hessian,
    :math:`\frac{\partial^2 \text{SSE}}{\partial \boldsymbol{\theta}^2}`
    or
    :math:`\frac{\partial^2 \text{WSSE}}{\partial \boldsymbol{\theta}^2}`, and
    applying Gauss-Newton approximation which results in:

    .. math::
       \boldsymbol{V}_{\boldsymbol{\theta}} = \left(\sum_{i = 1}^n \boldsymbol{G}_{i}^{\text{T}}
        \boldsymbol{\Sigma}_{\boldsymbol{y}}^{-1} \boldsymbol{G}_{i} \right)^{-1}

    where

    .. math::
       \boldsymbol{G}_{i} = \frac{\partial \boldsymbol{f}(\boldsymbol{x}_i;\boldsymbol{\theta})}
        {\partial \boldsymbol{\theta}}

    This method uses central finite difference to compute the Jacobian matrix, :math:`\boldsymbol{G}_{i}`,
    for experiment :math:`i`.

    .. math::
       \boldsymbol{G}_{i}[:,\,k] \approx \frac{\boldsymbol{f}(\boldsymbol{x}_i;\theta_k + \Delta \theta_k)
        \vert_{\hat{\boldsymbol{\theta}}} - \boldsymbol{f}(\boldsymbol{x}_i;\theta_k - \Delta \theta_k)
        \vert_{\hat{\boldsymbol{\theta}}}}{2 \Delta \theta_k} \quad \forall \quad \theta_k \, \in \,
        [\theta_1,\cdots, \theta_p]

3. Automatic Differentiation Method

    Similar to the finite difference method, the covariance matrix is calculated as:

    .. math::
       \boldsymbol{V}_{\boldsymbol{\theta}} = \left( \sum_{i = 1}^n \boldsymbol{G}_{\text{kaug},\, i}^{\text{T}}
        \boldsymbol{\Sigma}_{\boldsymbol{y}}^{-1} \boldsymbol{G}_{\text{kaug},\, i} \right)^{-1}

    However, this method uses implicit differentiation and the model-optimality or Karush–Kuhn–Tucker (KKT) conditions
    to compute the Jacobian matrix, :math:`\boldsymbol{G}_{\text{kaug},\, i}`, for experiment :math:`i`.

    .. math::
       \boldsymbol{G}_{\text{kaug},\,i} = \frac{\partial \boldsymbol{f}(\boldsymbol{x}_i,\boldsymbol{\theta})}
        {\partial \boldsymbol{\theta}} + \frac{\partial \boldsymbol{f}(\boldsymbol{x}_i,\boldsymbol{\theta})}
        {\partial \boldsymbol{x}_i}\frac{\partial \boldsymbol{x}_i}{\partial \boldsymbol{\theta}}

The covariance matrix calculation is only supported with the built-in objective functions "SSE" or "SSE_weighted".

In parmest, the covariance matrix can be computed after creating the
:class:`~pyomo.contrib.parmest.experiment.Experiment` class,
defining the :class:`~pyomo.contrib.parmest.parmest.Estimator` object,
and estimating the model parameters using :class:`~pyomo.contrib.parmest.parmest.Estimator.theta_est`
(all these steps were addressed in the :ref:`driversection` Section).

To estimate the covariance matrix, with the default method being "finite_difference", call
the :class:`~pyomo.contrib.parmest.parmest.Estimator.cov_est` function as follows:

.. testsetup:: *
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    # Data
    import pandas as pd
    data = pd.DataFrame(
        data=[[1, 8.3], [2, 10.3], [3, 19.0],
              [4, 16.0], [5, 15.6], [7, 19.8]],
        columns=['hour', 'y'],
    )

    # Create the Experiment class
    from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import RooneyBieglerExperiment

    exp_list = []
    for i in range(data.shape[0]):
        exp_list.append(RooneyBieglerExperiment(data.loc[i, :]))

.. doctest::
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    >>> import pyomo.contrib.parmest.parmest as parmest
    >>> pest = parmest.Estimator(exp_list, obj_function="SSE")
    >>> obj_val, theta_val = pest.theta_est()
    >>> cov = pest.cov_est()

Optionally, one of the three methods; "reduced_hessian", "finite_difference",
and "automatic_differentiation_kaug" can be supplied for the covariance calculation,
e.g.,

.. doctest::
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    >>> pest = parmest.Estimator(exp_list, obj_function="SSE")
    >>> obj_val, theta_val = pest.theta_est()
    >>> cov_method = "reduced_hessian"
    >>> cov = pest.cov_est(method=cov_method)