Covariance Matrix Estimation
=================================

The uncertainty in the estimated parameters is quantified using the covariance matrix.
The diagonal of the covariance matrix contains the variance of the estimated parameters.
Assuming Gaussian independent and identically distributed measurement errors, the
covariance matrix of the estimated parameters can be computed using the following
methods which have been implemented in parmest.

1. Reduced Hessian Method

    When the objective function is the sum of squared errors (SSE) between the
    observed and predicted values of the measured variables, the covariance matrix is:

    .. math::
       V_{\boldsymbol{\theta}} = 2 \sigma^2 \left(\frac{\partial^2 \text{SSE}}
        {\partial \boldsymbol{\theta} \partial \boldsymbol{\theta}}\right)^{-1}_{\boldsymbol{\theta}
        = \boldsymbol{\theta}^*}

    When the objective function is the weighted SSE (WSSE), the covariance matrix is:

    .. math::
       V_{\boldsymbol{\theta}} = \left(\frac{\partial^2 \text{WSSE}}
        {\partial \boldsymbol{\theta} \partial \boldsymbol{\theta}}\right)^{-1}_{\boldsymbol{\theta}
        = \boldsymbol{\theta}^*}

    Where :math:`V_{\boldsymbol{\theta}}` is the covariance matrix of the estimated
    parameters, :math:`\boldsymbol{\theta}` are the unknown parameters,
    :math:`\boldsymbol{\theta^*}` are the estimates of the unknown parameters, and
    :math:`\sigma^2` is the variance of the measurement error. When the standard
    deviation of the measurement error is not supplied by the user, parmest
    approximates the variance of the measurement error as
    :math:`\sigma^2 = \frac{1}{n-l} \sum e_i^2` where :math:`n` is the number of data
    points, :math:`l` is the number of fitted parameters, and :math:`e_i` is the
    residual for experiment :math:`i`.

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

    # Create the Rooney-Biegler model
    def rooney_biegler_model():
        """
        Formulates the Pyomo model of the Rooney-Biegler example

        Returns:
            m: Pyomo model
        """
        m = pyo.ConcreteModel()

        m.asymptote = pyo.Var(within=pyo.NonNegativeReals, initialize=10)
        m.rate_constant = pyo.Var(within=pyo.NonNegativeReals, initialize=0.2)

        m.hour = pyo.Var(within=pyo.PositiveReals, initialize=0.1)
        m.y = pyo.Var(within=pyo.NonNegativeReals)

        @m.Constraint()
        def response_rule(m):
            return m.y == m.asymptote * (1 - pyo.exp(-m.rate_constant * m.hour))

        return m

    # Create the Experiment class
    from pyomo.contrib.parmest.experiment import Experiment
    class RooneyBieglerExperiment(Experiment):
        def __init__(self, hour, y):
            self.y = y
            self.hour = hour
            self.model = None

        def get_labeled_model(self):
            self.create_model()
            self.finalize_model()
            self.label_model()

            return self.model

        def create_model(self):
            m = self.model = rooney_biegler_model()

            return m

        def finalize_model(self):
            m = self.model

            # fix the input variable
            m.hour.fix(self.hour)

            return m

        def label_model(self):
            m = self.model

            # add experiment outputs
            m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.experiment_outputs.update([(m.y, self.y)])

            # add unknown parameters
            m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.unknown_parameters.update(
                (k, pyo.value(k)) for k in [m.asymptote, m.rate_constant]
            )

            # create the measurement error
            m.measurement_error = pyo.Suffix(direction = pyo.Suffix.LOCAL)
            m.measurement_error.update([(m.y, None)])

    # Create an experiment list
    exp_list = []
    for i in range(data.shape[0]):
        exp_list.append(RooneyBieglerExperiment(data["hour"][i], data["y"][i]))

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