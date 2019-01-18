Overview
================

The Python package called parmest facilitates model-based
parameter estimation along with characterization of 
uncertainty associated with the estimates. For example, parmest
can provide confidence regions around the parameter estimates.
Additionally, parameter vectors, each with an attached probability estimate,
can be used to build scenarios for design optimization.

Functionality in parmest includes:

* Model based parameter estimation using experimental data
* Bootstrap resampling for parameter estimation
* Confidence regions based on single or multi-variate distributions
* Likelihood ratio
* Parallel processing

Background
----------

The goal of parameter estimation is to estimate values for 
a vector, :math:`{\theta}`, to use in the functional form

.. math::
      
   y = g(x; \theta)

where :math:`x` is a vector containing measured data, typically in high dimension, :math:`{\theta}` is 
a vector of values to estimate, in much lower dimension, and the response vectors are 
given as :math:`y_{i}, i=1,\ldots,m` with :math:`m` also much
smaller than the dimension of :math:`x`.  This is done by collecting :math:`S` data points, which
are :math:`{\tilde{x}},{\tilde{y}}` pairs and then finding :math:`{\theta}` values that 
minimize some function of the deviation between the values of :math:`{\tilde{y}}` that are measured
and the values of :math:`g({\tilde{x}};{\theta})` for each corresponding :math:`{\tilde{x}}`,
which is a subvector of the vector :math:`x`. Note
that for most experiments, only small parts of :math:`x` will change from
one experiment to the next.

The following least squares objective can be used to estimate parameter values, where data points are indexed by :math:`s=1,\ldots,S`

.. math::

   \min_{{\theta}} Q({\theta};{\tilde{x}}, {\tilde{y}}) \equiv \sum_{s=1}^{S}q_{s}({\theta};{\tilde{x}}_{s}, {\tilde{y}}_{s}) \;\;

where

.. math::

   q_{s}({\theta};{\tilde{x}}_{s}, {\tilde{y}}_{s}) = \sum_{i=1}^{m}w_{i}\left[{\tilde{y}}_{si} - g_{i}({\tilde{x}}_{s};{\theta})\right]^{2}, 

i.e., the contribution of sample :math:`s` to :math:`Q`, where :math:`w \in \Re^{m}` is a vector
of weights for the responses. For multi-dimensional :math:`y`, this
is the squared weighted :math:`L_{2}` norm and for univariate :math:`y` the weighted squared deviation.
Custom objectives can also be defined for parameter estimation.

In the applications of interest to us, the function :math:`g(\cdot)` is
usually defined as an optimization problem with a large number of
(perhaps constrained) optimization variables, a subset of which are
fixed at values :math:`{\tilde{x}}` when the optimization is performed. 
In other applications, the values of
:math:`{\theta}` are fixed parameter values, but for the problem formulation above,
the values of :math:`{\theta}` are the primary optimization variables. Note
that in general, the function :math:`g(\cdot)` will have a large set of
parameters that are not included in :math:`{\theta}`. Often, the :math:`y_{is}` will
be vectors themselves, perhaps indexed by time with index sets
that vary with :math:`s`.
