.. _page-data-from-dict:

Using a Python Dictionary
=========================

Data can be passed to the model
:meth:`~pyomo.environ.AbstractModel.create_instance` method
through a series of nested native Python dictionaries.  The structure
begins with a dictionary of *namespaces*, with the only required entry
being the ``None`` namespace.  Each namespace contains a dictionary that
maps component names to dictionaries of component values.  For scalar
components, the required data dictionary maps the implicit index
``None`` to the desired value:

 .. doctest::

    >>> from pyomo.environ import *
    >>> m = AbstractModel()
    >>> m.I = Set()
    >>> m.p = Param()
    >>> m.q = Param(m.I)
    >>> m.r = Param(m.I, m.I, default=0)
    >>> data = {None: {
    ...     'I': {None: [1,2,3]},
    ...     'p': {None: 100},
    ...     'q': {1: 10, 2:20, 3:30},
    ...     'r': {(1,1): 110, (1,2): 120, (2,3): 230},
    ... }}
    >>> i = m.create_instance(data)
    >>> i.pprint()
    2 Set Declarations
        I : Dim=0, Dimen=1, Size=3, Domain=None, Ordered=False, Bounds=(1, 3)
            [1, 2, 3]
        r_index : Dim=0, Dimen=2, Size=9, Domain=None, Ordered=False, Bounds=None
            Virtual
    <BLANKLINE>
    3 Param Declarations
        p : Size=1, Index=None, Domain=Any, Default=None, Mutable=False
            Key  : Value
            None :   100
        q : Size=3, Index=I, Domain=Any, Default=None, Mutable=False
            Key : Value
              1 :    10
              2 :    20
              3 :    30
        r : Size=9, Index=r_index, Domain=Any, Default=0, Mutable=False
            Key    : Value
            (1, 1) :   110
            (1, 2) :   120
            (2, 3) :   230
    <BLANKLINE>
    5 Declarations: I p q r_index r


