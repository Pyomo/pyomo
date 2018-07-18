#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import sys
import pprint as _pprint_

from pyomo.core.expr.numvalue import \
    NumericValue
from pyomo.core.kernel.component_interface import \
    (ICategorizedObject,
     _ActiveObjectMixin)

import six

def pprint(obj, indent=0, stream=sys.stdout):
    """pprint a kernel modeling object"""
    # ugly hack for ctypes
    import pyomo.core.base
    if not isinstance(obj, ICategorizedObject):
        if isinstance(obj, NumericValue):
            prefix = ""
            if indent > 0:
                prefix = (" "*indent)+" - "
            stream.write(prefix+str(obj)+"\n")
        else:
            assert indent == 0
            _pprint_.pprint(obj, indent=indent+1, stream=stream)
        return
    if not obj._is_component:
        # a container but not a block
        assert obj._is_container
        prefix = ""
        if indent > 0:
            prefix = (" "*indent)+" - "
        if isinstance(obj, _ActiveObjectMixin):
            stream.write(prefix+"%s: container(size=%s, active=%s, ctype=%s)\n"
                  % (str(obj), len(obj), obj.active, obj.ctype.__name__))
        else:
            stream.write(prefix+"%s: container(size=%s, ctype=%s)\n"
                  % (str(obj), len(obj), obj.ctype.__name__))
        for c in obj.children():
            pprint(c, indent=indent+1, stream=stream)
    elif not obj._is_container:
        prefix = ""
        if indent > 0:
            prefix = (" "*indent)+" - "
        # not a block
        clsname = obj.__class__.__name__
        if obj.ctype is pyomo.core.base.Var:
            stream.write(prefix+"%s: %s(value=%s, bounds=(%s,%s), domain_type=%s, fixed=%s, stale=%s)\n"
                  % (str(obj),
                     clsname,
                     obj.value,
                     obj.lb,
                     obj.ub,
                     obj.domain_type.__name__,
                     obj.fixed,
                     obj.stale))
        elif obj.ctype is pyomo.core.base.Constraint:
              stream.write(prefix+"%s: %s(active=%s, expr=%s)\n"
                  % (str(obj),
                     clsname,
                     obj.active,
                     str(obj.expr)))
        elif obj.ctype is pyomo.core.base.Objective:
            stream.write(prefix+"%s: %s(active=%s, expr=%s)\n"
                  % (str(obj), clsname, obj.active, str(obj.expr)))
        elif obj.ctype is pyomo.core.base.Expression:
            stream.write(prefix+"%s: %s(expr=%s)\n"
                  % (str(obj), clsname, str(obj.expr)))
        elif obj.ctype is pyomo.core.base.Param:
            stream.write(prefix+"%s: %s(value=%s)\n"
                  % (str(obj), clsname, str(obj.value)))
        elif obj.ctype is pyomo.core.base.SOSConstraint:
            stream.write(prefix+"%s: %s(active=%s, level=%s, entries=%s)\n"
                  % (str(obj),
                     clsname,
                     obj.active,
                     obj.level,
                     str(["(%s,%s)" % (str(v), w)
                          for v,w in zip(obj.variables,
                                         obj.weights)])))
        else:
            assert obj.ctype is pyomo.core.base.Suffix
            stream.write(prefix+"%s: %s(size=%s)\n"
                  % (str(obj.name), clsname,str(len(obj))))
    else:
        # a block
        for i, block in enumerate(obj.blocks()):
            if i > 0:
                stream.write("\n")
            stream.write((" "*indent)+"block: %s\n" % (str(block)))
            ctypes = block.collect_ctypes(descend_into=False)
            for ctype in sorted(ctypes,
                                key=lambda x: str(x)):
                stream.write((" "*indent)+'ctype='+ctype.__name__+" declarations:\n")
                for c in block.children(ctype=ctype):
                    if ctype is pyomo.core.base.Block:
                        if c._is_component:
                            stream.write((" "*indent)+"  - %s: block(children=%s)\n"
                                         % (str(c), len(list(c.children()))))
                        else:
                            stream.write((" "*indent)+"  - %s: block_container(size=%s)\n"
                                  % (str(c), len(list(c))))

                    else:
                        pprint(c, indent=indent+1, stream=stream)
