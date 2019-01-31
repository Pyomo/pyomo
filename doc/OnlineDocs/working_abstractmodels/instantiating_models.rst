Instantiating Models
-------------------- 

If you start with a :class:`~pyomo.environ.ConcreteModel`, each component
you add to the model will be fully constructed and initialized at the
time it attached to the model.  However, if you are starting with an
:class:`~pyomo.environ.AbstractModel`, construction occurs in two
phases.  When you first declare and attach components to the model,
those components are empty containers and *not* fully constructed, even
if you explicitly provide data.

.. doctest::

   >>> import pyomo.environ as pyo
   >>> model = pyo.AbstractModel()
   >>> model.is_constructed()
   False

   >>> model.p = pyo.Param(initialize=5)
   >>> model.p.is_constructed()
   False

   >>> model.I = pyo.Set(initialize=[1,2,3])
   >>> model.x = pyo.Var(model.I)
   >>> model.x.is_constructed()
   False

If you look at the ``model`` at this point, you will see that everything
is "empty":

.. doctest::

   >>> model.pprint()
   1 Set Declarations
       I : Dim=0, Dimen=1, Size=0, Domain=None, Ordered=False, Bounds=None
           Not constructed
   <BLANKLINE>
   1 Param Declarations
       p : Size=0, Index=None, Domain=Any, Default=None, Mutable=False
           Not constructed
   <BLANKLINE>
   1 Var Declarations
       x : Size=0, Index=I
           Not constructed
   <BLANKLINE>
   3 Declarations: p I x

Before you can manipulate modeling components or solve the model, you
must first create a concrete `instance` by applying data to your
abstract model.  This can be done using the
:meth:`~pyomo.environ.AbstractModel.create_instance` method, which takes
the abstract model and optional data and returns a new `concrete`
instance by constructing each of the model components in the order in
which they were declared (attached to the model).  Note that the
instance creation is performed "out of place"; that is, the original
abstract ``model`` is left untouched.

.. doctest::

   >>> instance = model.create_instance()
   >>> model.is_constructed()
   False
   >>> type(instance)
   <class 'pyomo.core.base.PyomoModel.ConcreteModel'>
   >>> instance.is_constructed()
   True
   >>> instance.pprint()
   1 Set Declarations
       I : Dim=0, Dimen=1, Size=3, Domain=None, Ordered=False, Bounds=(1, 3)
           [1, 2, 3]
   <BLANKLINE>
   1 Param Declarations
       p : Size=1, Index=None, Domain=Any, Default=None, Mutable=False
           Key  : Value
           None :     5
   <BLANKLINE>
   1 Var Declarations
       x : Size=3, Index=I
           Key : Lower : Value : Upper : Fixed : Stale : Domain
             1 :  None :  None :  None : False :  True :  Reals
             2 :  None :  None :  None : False :  True :  Reals
             3 :  None :  None :  None : False :  True :  Reals
   <BLANKLINE>
   3 Declarations: p I x

.. note::

   AbstractModel users should note that in some examples, your concrete
   model instance is called "`instance`" and not "`model`". This
   is the case here, where we are explicitly calling
   ``instance = model.create_instance()``.

The :meth:`~pyomo.environ.AbstractModel.create_instance` method can also
take a reference to external data, which overrides any data specified in
the original component declarations.  The data can be provided from
several sources, including using a :ref:`dict <page-data-from-dict>`,
:ref:`DataPortal <page-dataportals>`, or :ref:`DAT file
<page-datfiles>`.  For example:

.. doctest::

   >>> instance2 = model.create_instance({None: {'I': {None: [4,5]}}})
   >>> instance2.pprint()
   1 Set Declarations
       I : Dim=0, Dimen=1, Size=2, Domain=None, Ordered=False, Bounds=(4, 5)
           [4, 5]
   <BLANKLINE>
   1 Param Declarations
       p : Size=1, Index=None, Domain=Any, Default=None, Mutable=False
           Key  : Value
           None :     5
   <BLANKLINE>
   1 Var Declarations
       x : Size=2, Index=I
           Key : Lower : Value : Upper : Fixed : Stale : Domain
             4 :  None :  None :  None : False :  True :  Reals
             5 :  None :  None :  None : False :  True :  Reals
   <BLANKLINE>
   3 Declarations: p I x
