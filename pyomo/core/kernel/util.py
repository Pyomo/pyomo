#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pprint as _pprint_

from pyomo.core.kernel.numvalue import \
    NumericValue
from pyomo.core.kernel.component_interface import \
    (ICategorizedObject,
     _ActiveComponentContainerMixin)

import six

def pprint(obj, indent=0):
    """pprint a kernel modeling object"""
    # ugly hack for ctypes
    import pyomo.core.base
    if not isinstance(obj, ICategorizedObject):
        if isinstance(obj, NumericValue):
            prefix = ""
            if indent > 0:
                prefix = (" "*indent)+" - "
            print(prefix+str(obj))
        else:
            assert indent == 0
            _pprint_.pprint(obj, indent=indent+1)
        return
    if not obj._is_component:
        # a container but not a block
        assert obj._is_container
        prefix = ""
        if indent > 0:
            prefix = (" "*indent)+" - "
        if isinstance(obj, _ActiveComponentContainerMixin):
            print(prefix+"%s: container(size=%s, active=%s, ctype=%s)"
                  % (str(obj), len(obj), obj.active, obj.ctype.__name__))
        else:
            print(prefix+"%s: container(size=%s, ctype=%s)"
                  % (str(obj), len(obj), obj.ctype.__name__))
        for c in obj.children():
            pprint(c, indent=indent+1)
    elif not obj._is_container:
        prefix = ""
        if indent > 0:
            prefix = (" "*indent)+" - "
        # not a block
        clsname = obj.__class__.__name__
        if obj.ctype is pyomo.core.base.Var:
            print(prefix+"%s: %s(value=%s, bounds=(%s,%s), domain_type=%s, fixed=%s, stale=%s)"
                  % (str(obj),
                     clsname,
                     obj.value,
                     obj.lb,
                     obj.ub,
                     obj.domain_type.__name__,
                     obj.fixed,
                     obj.stale))
        elif obj.ctype is pyomo.core.base.Constraint:
              print(prefix+"%s: %s(active=%s, expr=%s)"
                  % (str(obj),
                     clsname,
                     obj.active,
                     str(obj.expr)))
        elif obj.ctype is pyomo.core.base.Objective:
            print(prefix+"%s: %s(active=%s, expr=%s)"
                  % (str(obj), clsname, obj.active, str(obj.expr)))
        elif obj.ctype is pyomo.core.base.Expression:
            print(prefix+"%s: %s(expr=%s)"
                  % (str(obj), clsname, str(obj.expr)))
        elif obj.ctype is pyomo.core.base.Param:
            print(prefix+"%s: %s(value=%s)"
                  % (str(obj), clsname, str(obj.value)))
        elif obj.ctype is pyomo.core.base.SOSConstraint:
            print(prefix+"%s: %s(active=%s, level=%s, entries=%s)"
                  % (str(obj),
                     clsname,
                     obj.active,
                     obj.level,
                     str(["(%s,%s)" % (str(v), w)
                          for v,w in zip(obj.variables,
                                         obj.weights)])))
        else:
            assert obj.ctype is pyomo.core.base.Suffix
            print(prefix+"%s: %s(size=%s)"
                  % (str(obj.name), clsname,str(len(obj))))
    else:
        # a block
        for i, block in enumerate(obj.blocks()):
            if i > 0:
                print("")
            print((" "*indent)+"block: %s" % (str(block)))
            ctypes = block.collect_ctypes(descend_into=False)
            for ctype in sorted(ctypes,
                                key=lambda x: str(x)):
                print((" "*indent)+'ctype='+ctype.__name__+" declarations:")
                for c in block.children(ctype=ctype):
                    if ctype is pyomo.core.base.Block:
                        if c._is_component:
                            print((" "*indent)+"  - %s: block(children=%s)"
                                  % (str(c), len(list(c.children()))))
                        else:
                            print((" "*indent)+"  - %s: block_container(size=%s)"
                                  % (str(c), len(list(c))))

                    else:
                        pprint(c, indent=indent+1)
