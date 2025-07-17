Covariance Matrix Estimation
=================================

The uncertainty in the estimated parameters is quantified using the covariance matrix.
The diagonal of the covariance matrix contains the variance of the estimated parameters.
Assuming Gaussian independent and identically distributed measurement errors, the
covariance matrix of the estimated parameters can be computed using the following
methods which have been implemented in parmest.

1. Reduced Hessian Method

    .. math::
       V_{\boldsymbol{\theta}} = 2 \sigma^2 \left(\frac{\partial^2 \text{SSE}}
        {\partial \boldsymbol{\theta} \partial \boldsymbol{\theta}}\right)^{-1}_{\boldsymbol{\theta}
        = \boldsymbol{\theta}^*}

    Where SSE is the sum of squared errors, WSSE is the weighted SSE,
    :math:`\boldsymbol{\theta}` are the unknown parameters, :math:`\boldsymbol{\theta^*}`
    are the estimate of the unknown parameters, and :math:`\sigma^2` is the variance of
    the measurement error. When the standard deviation of the measurement error is not
    supplied by the user, parmest approximates the variance of the measurement error as
    :math:`\sigma^2 = \frac{1}{n-l} \sum e_i^2` where :math:`n` is the number of data
    points, :math:`l` is the number of fitted parameters, and :math:`e_i` is the residual
    for experiment :math:`i`.

2. Finite Difference Method

    .. math::
       V_{\boldsymbol{\theta}} = \left( \sum_{r = 1}^n \mathbf{G}_{r}^{\mathrm{T}} \mathbf{W}
        \mathbf{G}_{r} \right)^{-1}

    This method uses central finite difference to compute the Jacobian matrix,
    :math:`\mathbf{G}_{r}`, which is the sensitivity of the measured variables with
    respect to the parameters, `\boldsymbol{\theta}`. :math:`\mathbf{W}` is a diagonal
    matrix containing the inverse of the variance of the measurement errors,
    :math:`\sigma^2`.

3. Automatic Differentiation Method

    .. math::
       V_{\boldsymbol{\theta}} = \left( \sum_{r = 1}^n \mathbf{G}_{\text{kaug},\, r}^{\mathrm{T}}
        \mathbf{W} \mathbf{G}_{\text{kaug},\, r} \right)^{-1}

    This method uses the model optimality (KKT) condition to compute the Jacobian matrix,
    :math:`\mathbf{G}_{\text{kaug},\, r}`.

In parmest, the covariance matrix can be calculated after defining the
:class:`~pyomo.contrib.parmest.parmest.Estimator` object and estimating the unknown
parameters using :class:`~pyomo.contrib.parmest.parmest.Estimator.theta_est`. To
estimate the covariance matrix, call
:class:`~pyomo.contrib.parmest.parmest.Estimator.cov_est` and pass it the number
of data points, e.g.,

.. testsetup:: *
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    # Data
    import pandas as pd
    data = pd.DataFrame(
        data=[[1, 8.3], [2, 10.3], [3, 19.0],
              [4, 16.0], [5, 15.6], [7, 19.8]],
        columns=['hour', 'y'],
    )
    num_data = len(data)

    # Create an experiment list
    from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import RooneyBieglerExperiment
    exp_list = []
    for i in range(data.shape[0]):
        exp_list.append(RooneyBieglerExperiment(data.loc[i, :]))

.. doctest::
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    >>> import pyomo.contrib.parmest.parmest as parmest
    >>> pest = parmest.Estimator(exp_list, obj_function="SSE")
    >>> obj_val, theta_val = pest.theta_est()
    >>> cov = pest.cov_est(cov_n=num_data)

Optionally, one of the three methods; "reduced_hessian", "finite_difference",
and "automatic_differentiation_kaug" can be supplied for the covariance calculation,
e.g.,

.. doctest::
    :skipif: not __import__('pyomo.contrib.parmest.parmest').contrib.parmest.parmest.parmest_available

    >>> pest = parmest.Estimator(exp_list, obj_function="SSE")
    >>> obj_val, theta_val = pest.theta_est()
    >>> cov_method = "reduced_hessian"
    >>> cov = pest.cov_est(cov_n=num_data, method=cov_method)