Covariance Matrix Estimation
============================

The uncertainty in the estimated parameters is quantified using the covariance matrix.
The diagonal of the covariance matrix contains the variance of the estimated parameters.
Assuming Gaussian independent and identically distributed measurement errors, the
covariance matrix of the estimated parameters can be computed using the following
methods which have been implemented in parmest.

1. Reduced Hessian Method

    When the objective function is the sum of squared errors (SSE):
    :math:`\text{SSE} = \sum_{i = 1}^n \left(y_{i} - \hat{y}_{i}\right)^2`,
    the covariance matrix is:

    .. math::
       V_{\boldsymbol{\theta}} = 2 \sigma^2 \left(\frac{\partial^2 \text{SSE}}
        {\partial \boldsymbol{\theta} \partial \boldsymbol{\theta}}\right)^{-1}_{\boldsymbol{\theta}
        = \boldsymbol{\theta}^*}

    When the objective function is the weighted SSE (WSSE):
    :math:`\text{WSSE} = \frac{1}{2} \left(\mathbf{y} - f(\mathbf{x};\boldsymbol{\theta})\right)^\text{T}
    \mathbf{W} \left(\mathbf{y} - f(\mathbf{x};\boldsymbol{\theta})\right)`,
    the covariance matrix is:

    .. math::
       V_{\boldsymbol{\theta}} = \left(\frac{\partial^2 \text{WSSE}}
        {\partial \boldsymbol{\theta} \partial \boldsymbol{\theta}}\right)^{-1}_{\boldsymbol{\theta}
        = \boldsymbol{\theta}^*}

    Where :math:`V_{\boldsymbol{\theta}}` is the covariance matrix of the estimated
    parameters, :math:`y` are the observed measured variables, :math:`\hat{y}` are the
    predicted measured variables, :math:`n` is the number of data points,
    :math:`\boldsymbol{\theta}` are the unknown parameters, :math:`\boldsymbol{\theta^*}`
    are the estimates of the unknown parameters, :math:`\mathbf{x}` are the decision
    variables, and :math:`\mathbf{W}` is a diagonal matrix containing the inverse of the
    variance of the measurement error, :math:`\sigma^2`. When the standard
    deviation of the measurement error is not supplied by the user, parmest
    approximates the variance of the measurement error as
    :math:`\sigma^2 = \frac{1}{n-l} \sum e_i^2` where :math:`l` is the number of
    fitted parameters, and :math:`e_i` is the residual for experiment :math:`i`.

    In parmest, this method computes the inverse of the Hessian by scaling the
    objective function (SSE or WSSE) with a constant probability factor.

2. Finite Difference Method

    In this method, the covariance matrix, :math:`V_{\boldsymbol{\theta}}`, is
    calculated by applying the Gauss-Newton approximation to the Hessian,
    :math:`\frac{\partial^2 \text{SSE}}{\partial \boldsymbol{\theta} \partial \boldsymbol{\theta}}`
    or
    :math:`\frac{\partial^2 \text{WSSE}}{\partial \boldsymbol{\theta} \partial \boldsymbol{\theta}}`,
    leading to:

    .. math::
       V_{\boldsymbol{\theta}} = \left(\sum_{i = 1}^n \mathbf{G}_{i}^{\mathrm{T}} \mathbf{W}
        \mathbf{G}_{i} \right)^{-1}

    This method uses central finite difference to compute the Jacobian matrix,
    :math:`\mathbf{G}_{i}`, for experiment :math:`i`, which is the sensitivity of
    the measured variables with respect to the parameters, :math:`\boldsymbol{\theta}`.
    :math:`\mathbf{W}` is a diagonal matrix containing the inverse of the variance
    of the measurement errors, :math:`\sigma^2`.

3. Automatic Differentiation Method

    Similar to the finite difference method, the covariance matrix is calculated as:

    .. math::
       V_{\boldsymbol{\theta}} = \left( \sum_{i = 1}^n \mathbf{G}_{\text{kaug},\, i}^{\mathrm{T}}
        \mathbf{W} \mathbf{G}_{\text{kaug},\, i} \right)^{-1}

    However, this method uses the model optimality (KKT) condition to compute the
    Jacobian matrix, :math:`\mathbf{G}_{\text{kaug},\, i}`, for experiment :math:`i`.

The covariance matrix calculation is only supported with the built-in objective
functions "SSE" or "SSE_weighted".

In parmest, the covariance matrix can be calculated after defining the
:class:`~pyomo.contrib.parmest.parmest.Estimator` object and estimating the unknown
parameters using :class:`~pyomo.contrib.parmest.parmest.Estimator.theta_est`. To
estimate the covariance matrix, with the default method being "finite_difference", call
the :class:`~pyomo.contrib.parmest.parmest.Estimator.cov_est` function, e.g.,

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