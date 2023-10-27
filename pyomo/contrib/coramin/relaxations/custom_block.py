import sys
from pyomo.core.base.block import Block
from pyomo.core.base.indexed_component import UnindexedComponent_set
# ToDo: documentation
# ToDo: passing of kwargs down to the data object
# ToDo: figure out if the setattr's are necessary in the decorator
# ToDo: decide if we need the decorator (it actually does not do much
#       and can be replaced by one more class declaration that might be "easier"
#       for the user anyway?
'''
This module implements meta classes and a decorator to make
it easier to create derived block types. With the decorator,
you only need to inherit from _BlockData.

# ToDo: Document this custom block code with an example
'''

class _IndexedCustomBlockMeta(type):
    """Metaclass for creating an indexed block with
    a custom block data type."""

    def __new__(meta, name, bases, dct):
        def __init__(self, *args, **kwargs):
            bases[0].__init__(self, *args, **kwargs)
        dct["__init__"] = __init__
        return type.__new__(meta, name, bases, dct)

class _ScalarCustomBlockMeta(type):
    '''Metaclass used to create a scalar block with a
    custom block data type
    '''
    def __new__(meta, name, bases, dct):
        def __init__(self, *args, **kwargs):
            # bases[0] is the custom block data object
            bases[0].__init__(self, component=self)
            # bases[1] is the custom block object that
            # is used for declaration
            bases[1].__init__(self, *args, **kwargs)
        dct["__init__"] = __init__
        return type.__new__(meta, name, bases, dct)

class CustomBlock(Block):
    ''' This CustomBlock is the base class that allows
    for easy creation of specialized derived blocks
    '''
    def __new__(cls, *args, **kwds):
        if cls.__name__.startswith('_Indexed') or \
                cls.__name__.startswith('_Scalar'):
            # we are entering here the second time (recursive)
            # therefore, we need to create what we have
            return super(CustomBlock, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args)==1):
            bname = "_Scalar{}".format(cls.__name__)
            n = _ScalarCustomBlockMeta(bname, (cls._ComponentDataClass, cls),{})
            return n.__new__(n)
        else:
            bname = "_Indexed{}".format(cls.__name__)
            n = _IndexedCustomBlockMeta(bname, (cls,), {})
            return n.__new__(n)


def declare_custom_block(name):
    ''' Decorator to declare the custom component
    that goes along with a custom block data

    @declare_custom_block(name=FooBlock)
    class FooBlockData(_BlockData):
       # custom block data class
    '''
    def proc_dec(cls):
        # this is the decorator function that
        # creates the block component class
        c = type(
            name, # name of new class
            (CustomBlock,), # base classes
            {"__module__": cls.__module__, "_ComponentDataClass": cls}) # magic to fix the module

        # are these necessary?
        setattr(sys.modules[cls.__module__], name, c)
        setattr(cls, '_orig_name', name)
        setattr(cls, '_orig_module', cls.__module__)
        return cls
    return proc_dec
