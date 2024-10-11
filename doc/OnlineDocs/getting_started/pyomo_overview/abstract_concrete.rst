Abstract Versus Concrete Models
-------------------------------

A mathematical model can be defined using symbols that represent data
values.  For example, the following equations represent a linear program
(LP) to find optimal values for the vector :math:`x` with parameters
:math:`n` and :math:`b`, and parameter vectors :math:`a` and :math:`c`:

.. math::
   :nowrap:

    \begin{array}{lll}
    \min       & \sum_{j=1}^n c_j x_j &\\
     \mathrm{s.t.} & \sum_{j=1}^n a_{ij} x_j \geq b_i & \forall i = 1 \ldots m\\
               & x_j \geq 0 & \forall j = 1 \ldots n
     \end{array}

.. note::

   As a convenience, we use the symbol :math:`\forall` to mean "for all"
   or "for each."

We call this an *abstract* or *symbolic* mathematical model since it
relies on unspecified parameter values.  Data values can be used to
specify a *model instance*.  The ``AbstractModel`` class provides a
context for defining and initializing abstract optimization models in
Pyomo when the data values will be supplied at the time a solution is to
be obtained.

In many contexts, a mathematical model can and should be directly
defined with the data values supplied at the time of the model
definition.  We call these *concrete* mathematical models.  For example,
the following LP model is a concrete instance of the previous abstract
model:

.. math::
   :nowrap:

    \begin{array}{ll}
    \min       & 2 x_1 + 3 x_2\\
     \mathrm{s.t.} & 3 x_1 + 4 x_2 \geq 1\\
               & x_1, x_2 \geq 0
    \end{array}

The ``ConcreteModel`` class is used to define concrete optimization
models in Pyomo.

.. note::

   Python programmers will probably prefer to write concrete models,
   while users of some other algebraic modeling languages may tend to
   prefer to write abstract models.  The choice is largely a matter of
   taste; some applications may be a little more straightforward using
   one or the other.
