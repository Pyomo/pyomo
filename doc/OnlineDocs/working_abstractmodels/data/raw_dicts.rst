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
        I : Size=1, Index=None, Ordered=Insertion
            Key  : Dimen : Domain : Size : Members
            None :     1 :    Any :    3 : {1, 2, 3}
        r_index : Size=1, Index=None, Ordered=True
            Key  : Dimen : Domain : Size : Members
            None :     2 :    I*I :    9 : {(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)}
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


