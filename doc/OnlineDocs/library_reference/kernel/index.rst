.. role:: python(code)
   :language: python

.. warning::

   The :python:`pyomo.kernel` API is still in the beta phase of development. It is fully tested and functional; however, the interface may change as it becomes further integrated with the rest of Pyomo.

.. warning::

   Models built with :python:`pyomo.kernel` components are not yet compatible with pyomo extension modules (e.g., :python:`pyomo.pysp`, :python:`pyomo.dae`, :python:`pyomo.gdp`).

The Kernel Library
==================

The :python:`pyomo.kernel` library is an experimental modeling interface designed to provide a better experience for users doing concrete modeling and advanced application development with Pyomo. It includes the basic set of :ref:`modeling components <kernel_modeling_components>` necessary to build algebraic models, which have been redesigned from the ground up to make it easier for users to customize and extend. For a side-by-side comparison of :python:`pyomo.kernel` and :python:`pyomo.environ` syntax, visit the link below.

.. toctree::

   syntax_comparison.rst


Models built from :python:`pyomo.kernel` components are fully compatible with the standard solver interfaces included with Pyomo. A minimal example script that defines and solves a model is shown below.

.. literalinclude:: examples/kernel_solving.py
   :language: python

Notable Improvements
--------------------

More Control of Model Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Containers in :python:`pyomo.kernel` are analogous to indexed components in :python:`pyomo.environ`. However, :python:`pyomo.kernel` containers allow for additional layers of structure as they can be nested within each other as long as they have compatible categories. The following example shows this using :python:`pyomo.kernel.variable` containers.

.. literalinclude:: examples/kernel_containers_all.spy
   :language: python

As the next section will show, the standard modeling component containers are also compatible with user-defined classes that derive from the existing modeling components.

Sub-Classing
^^^^^^^^^^^^

The existing components and containers in :python:`pyomo.kernel` are designed to make sub-classing easy. User-defined classes that derive from the standard modeling components and containers in :python:`pyomo.kernel` are compatible with existing containers of the same component category. As an example, in the following code we see that the :python:`pyomo.kernel.block_list` container can store both :python:`pyomo.kernel.block` objects as well as a user-defined :python:`Widget` object that derives from :python:`pyomo.kernel.block`. The :python:`Widget` object can also be placed on another block object as an attribute and treated itself as a block.

.. code-block:: python

   class Widget(pyomo.kernel.block):
       ...

   model = pyomo.kernel.block()
   model.blist = pyomo.kernel.block_list()
   model.blist.append(Widget())
   model.blist.append(pyomo.kernel.block())
   model.w = Widget()
   model.w.x = pyomo.kernel.variable()

The next series of examples goes into more detail on how to implement derived components or containers.

The following code block shows a class definition for a non-negative variable, starting from :python:`pyomo.kernel.variable` as a base class.

.. literalinclude:: examples/kernel_subclassing_Nonnegative.spy
   :language: python

The :python:`NonNegativeVariable` class prevents negative values from being stored into its lower bound during initialization or later on through assignment statements (e.g, :python:`x.lb = -1` fails). Note that the :python:`__slots__ == ()` line at the beginning of the class definition is optional, but it is recommended if no additional data members are necessary as it reduces the memory requirement of the new variable type.

The next code block defines a custom variable container called :python:`Point` that represents a 3-dimensional point in Cartesian space. The new type derives from the :python:`pyomo.kernel.variable_tuple` container and uses the :python:`NonNegativeVariable` type we defined previously in the `z` coordinate.

.. literalinclude:: examples/kernel_subclassing_Point.spy
   :language: python

The :python:`Point` class can be treated like a tuple storing three variables, and it can be placed inside of other variable containers or added as attributes to blocks. The property methods included in the class definition provide an additional syntax for accessing the three variables it stores, as the next code example will show.

The following code defines a class for building a convex second-order cone constraint from a :python:`Point` object. It derives from the :python:`pyomo.kernel.constraint` class, overriding the constructor to build the constraint expression and utilizing the property methods on the point class to increase readability.

.. literalinclude:: examples/kernel_subclassing_SOC.spy
   :language: python


Reduced Memory Usage
^^^^^^^^^^^^^^^^^^^^

The :python:`pyomo.kernel` library offers significant opportunities to reduce memory requirements for highly structured models. The situation where this is most apparent is when expressing a model in terms of many small blocks consisting of singleton components. As an example, consider expressing a model consisting of a large number of voltage transformers. One option for doing so might be to define a `Transformer` component as a subclass of :python:`pyomo.kernel.block`. The example below defines such a component, including some helper methods for connecting input and output voltage variables and updating the transformer ratio.

.. literalinclude:: examples/transformer_kernel.spy
   :language: python

A simplified version of this using :python:`pyomo.environ` components might look like what is below.

.. literalinclude:: examples/transformer_aml.spy
   :language: python

The transformer expressed using :python:`pyomo.kernel` components requires roughly 2 KB of memory, whereas the :python:`pyomo.environ` version requires roughly 8.4 KB of memory (an increase of more than 4x). Additionally, the :python:`pyomo.kernel` transformer is fully compatible with all existing :python:`pyomo.kernel` block containers.

Direct Support For Conic Constraints with Mosek
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pyomo 5.6.3 introduced support into :python:`pyomo.kernel`
for six conic constraint forms that are directly recognized
by the new Mosek solver interface. These are

 - :python:`conic.quadratic`:

     :math:`\;\;\sum_{i}x_i^2 \leq r^2,\;\;r\geq 0`

 - :python:`conic.rotated_quadratic`:

     :math:`\;\;\sum_{i}x_i^2 \leq 2 r_1 r_2,\;\;r_1,r_2\geq 0`

 - :python:`conic.primal_exponential`:

     :math:`\;\;x_1\exp(x_2/x_1) \leq r,\;\;x_1,r\geq 0`

 - :python:`conic.primal_power` (:math:`\alpha` is a constant):

     :math:`\;\;||x||_2 \leq r_1^{\alpha} r_2^{1-\alpha},\;\;r_1,r_2\geq 0,\;0 < \alpha < 1`

 - :python:`conic.dual_exponential`:

     :math:`\;\;-x_2\exp((x_1/x_2)-1) \leq r,\;\;x_2\leq0,\;r\geq 0`

 - :python:`conic.dual_power` (:math:`\alpha` is a constant):

     :math:`\;\;||x||_2 \leq (r_1/\alpha)^{\alpha} (r_2/(1-\alpha))^{1-\alpha},\;\;r_1,r_2\geq 0,\;0 < \alpha < 1`

Other solver interfaces will treat these objects as general
nonlinear or quadratic constraints, and may or may not have
the ability to identify their convexity. For instance,
Gurobi will recognize the expressions produced by the
:python:`quadratic` and :python:`rotated_quadratic` objects
as representing convex domains as long as the variables
involved satisfy the convexity conditions. However, other
solvers may not include this functionality.

Each of these conic constraint classes are of the same
category type as standard :python:`pyomo.kernel.constraint`
object, and, thus, are directly supported by the standard
constraint containers (:python:`constraint_tuple`,
:python:`constraint_list`, :python:`constraint_dict`).

Each conic constraint class supports two methods of
instantiation. The first method is to directly instantiate a
conic constraint object, providing all necessary input
variables:

.. literalinclude:: examples/conic_Class.spy
   :language: python

This method may be limiting if utilizing the Mosek solver as
the user must ensure that additional conic constraints do
not use variables that are directly involved in any existing
conic constraints (this is a limitation the Mosek solver
itself).

To overcome this limitation, and to provide a more general
way of defining conic domains, each conic constraint class
provides the :python:`as_domain` class method. This
alternate constructor has the same argument signature as the
class, but in place of each variable, one can optionally
provide a constant, a linear expression, or
:python:`None`. The :python:`as_domain` class method returns
a :python:`block` object that includes the core conic
constraint, auxiliary variables used to express the conic
constraint, as well as auxiliary constraints that link the
inputs (that are not :python:`None`) to the auxiliary
variables. Example:

.. literalinclude:: examples/conic_Domain.spy
   :language: python

Reference
---------

.. _kernel_modeling_components:

Modeling Components:
^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   block.rst
   variable.rst
   constraint.rst
   parameter.rst
   objective.rst
   expression.rst
   sos.rst
   suffix.rst
   piecewise/index.rst
   conic.rst

Base API:
^^^^^^^^^

.. toctree::
   :maxdepth: 1

   base.rst
   homogeneous_container.rst
   heterogeneous_container.rst

Containers:
^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   tuple_container.rst
   list_container.rst
   dict_container.rst
