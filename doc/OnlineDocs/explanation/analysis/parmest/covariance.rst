Covariance Matrix Estimation
=================================

If the optional argument ``calc_cov=True`` is specified for :class:`~pyomo.contrib.parmest.parmest.Estimator.theta_est`, 
parmest will calculate the covariance matrix :math:`V_{\theta}` as follows:

.. math::
   V_{\theta} = 2 \sigma^2 H^{-1} 

This formula assumes all measurement errors are independent and identically distributed with 
variance :math:`\sigma^2`. :math:`H^{-1}` is the inverse of the Hessian matrix for an unweighted 
sum of least squares problem. Currently, the covariance approximation is only valid if the 
objective given to parmest is the sum of squared error. Moreover, parmest approximates the 
variance of the measurement errors as :math:`\sigma^2 = \frac{1}{n-l} \sum e_i^2` where :math:`n` is 
the number of data points, :math:`l` is the number of fitted parameters, and :math:`e_i` is the 
residual for experiment :math:`i`.