#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.kernel.dict_container import DictContainer
from pyomo.core.kernel.tuple_container import TupleContainer
from pyomo.core.kernel.list_container import ListContainer

import six

def define_homogeneous_container_type(namespace,
                                      name,
                                      container_class,
                                      ctype,
                                      doc=None,
                                      use_slots=True):
    """
    This function is designed to be called for the simple
    container implementations (DictContainer, TupleContainer,
    and ListContainer) as the container_class argument.

    When called using the globals() namespace within a module, it
    is equivalent to placing the following class definition
    within that module:

        class <name>(<container_class>):
            <doc>
            _ctype = <ctype>

            ### if ### <use_slots>
            __slots__ = ["_parent",
                         "_storage_key",
                         "_active",
                         "_data"]
            if six.PY3:
                # Prior to Python 3, the abc module does not
                # use empty __slots__ declarations on the
                # base classes. Therefore, we do not need a
                # __weakref__ slot because there is already
                # a __dict__ in the class hierarchy.
                __slots__.append("__weakref__")
            ### fi ###

            def __init__(self, *args, **kwds):
                self._parent = None
                self._storage_key = None
                self._active = True
                super(<name>, self).__init__(*args, **kwds)
    """
    assert name not in namespace
    cls_dict = {}
    cls_dict['_ctype'] = ctype
    if use_slots:
        cls_dict['__slots__'] = ["_parent",
                                 "_storage_key",
                                 "_active",
                                 "_data"]
        if six.PY3:
            cls_dict['__slots__'].append("__weakref__")
        cls_dict['__slots__'] = tuple(cls_dict['__slots__'])

    def _init(self, *args, **kwds):
        self._parent = None
        self._storage_key = None
        self._active = True
        container_class.__init__(self, *args, **kwds)
    cls_dict['__init__'] = _init
    cls_dict['__module__'] = namespace['__name__']
    if doc is not None:
        cls_dict['__doc__'] = doc
    namespace[name] = type(name,
                           (container_class,),
                           cls_dict)

def define_simple_containers(namespace,
                             prefix,
                             ctype,
                             use_slots=True):
    """Use this function to define all three simple
    container definitions for a new object category type."""
    doc_ = ("A %s-style container for objects "
            "with category type "+ctype.__name__)
    for suffix, container_class in (('tuple', TupleContainer),
                                    ('list', ListContainer),
                                    ('dict', DictContainer)):
        doc = doc_ % (suffix,)
        define_homogeneous_container_type(namespace,
                                          prefix+"_"+suffix,
                                          container_class,
                                          ctype,
                                          doc=doc,
                                          use_slots=use_slots)
