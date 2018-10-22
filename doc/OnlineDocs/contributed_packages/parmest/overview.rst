Overview
================

The software called parmest is intended to allow for model-based
parameter estimation along with some characterization of the
uncertainty associated with the estimates. For example, parmest
can provide confidence regions around the parameter estimates.

Most of the parmest software deals with the fact that parameter values
estimates are typically based on data from experiments. 
The ultimate goal of parmest is to provide scenarios, which are
parameter vectors, :math:`{\theta}`, each with an attached probability estimate.
These scenarios are intended to be used in design optimization, either
directly, or as a way to check designs.

To use parmest, the user defines a PySP [PyomoBookII]_ callback function.  The 
callback function takes a scenario name and returns a populated 
and initialized model for that scenario. The objective
function needs to be given in PySP form with an ``Expression`` for
first and second stage costs: the first stage cost is set to zero 
while the second stage cost is the sum of squared deviation of the 
model :math:`y` values from the observed :math:`y` values.

Background
----------

Using parsimonious notation, the goal is to estimate values for 
a vector, :math:`{\theta}`, to use in a functional form

.. math::
      
   y = g(x; \theta)

where :math:`x` is a vector, typically in high dimension, :math:`{\theta}` is 
a vector in much lower dimension, :math:`p`, and the response vectors are 
given as :math:`y_{i}, i=1,\ldots,m` with :math:`m` also much
smaller than the dimension of :math:`x`.  This is done by collecting :math:`S` data points, which
are :math:`{\tilde{x}},{\tilde{y}}` pairs and then finding :math:`{\theta}` values that 
minimize some function of the deviation between the values of :math:`{\tilde{y}}` that are measured
and the values of :math:`g({\tilde{x}};{\theta})` for each corresponding :math:`{\tilde{x}}`,
which is a subvector of the vector :math:`x`. Note
that for most experiments, only small parts of :math:`x` will change from
one experiment to the next.

To be concrete, we start by assuming that the data points are indexed by :math:`s=1,\ldots,S`
and the parameters are fit using

.. math::

   \min_{{\theta}} Q({\theta};{\tilde{x}}, {\tilde{y}}) \equiv \sum_{s=1}^{S}q_{s}({\theta};{\tilde{x}}_{s}, {\tilde{y}}_{s}) \;\; (LSQ)

where

.. math::

   q_{s}({\theta};{\tilde{x}}_{s}, {\tilde{y}}_{s}) = \sum_{i=1}^{m}w_{i}\left[{\tilde{y}}_{si} - g_{i}({\tilde{x}}_{s};{\theta})\right]^{2}, 

i.e., the contribution of sample :math:`s` to :math:`Q`, where :math:`w \in \Re^{m}` is a vector
of weights for the responses. For multi-dimensional :math:`y`, this
is the squared weighted :math:`L_{2}` norm and for univariate :math:`y` the weighted squared deviation.

Other M-estimators of :math:`{\theta}` might be implemented in the future.
Aside: for various reasons one
might be happiest if the function given is divided by :math:`S` but that does not affect the minimization.

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
