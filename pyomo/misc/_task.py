"""
Defining a Coopr-specific task class
"""

__all__ = ['coopr_api', 'ICooprTask', 'CooprAPIFactory', 'CooprAPIData']

import copy
import inspect
import os
import logging
from six import iteritems, with_metaclass

import pyutilib.workflow
import pyutilib.misc
from coopr.core import plugin


logger = logging.getLogger('coopr.core')


class CooprAPIData(dict):
    """
    A generalization of pyutilib.misc.Bunch.  This class counts access to attributes, and
    it generates errors for undefined attributes.
    """

    def __init__(self, **kw):
        dict.__init__(self,kw)
        self.__dict__.update(kw)
        self._declared_ = set()
        self.clean()

    def clean(self):
        self._dirty_ = set()

    def unused(self):
        for k, v in self.__dict__.items():
            if not k in self._dirty_:
                yield k

    def declare(self, args):
        if type(args) in (list, tuple, set):
            self._declared_.update(args)
        else:
            self._declared_.add(args)

    def update(self, d):
        self.__dict__.update(d)

    def __setitem__(self, name, val):
        self.__setattr__(name,val)

    def __getitem__(self, name):
        return self.__getattr__(name)

    def __setattr__(self, name, val):
        if name[0] != '_':
            if len(self._declared_) > 0 and not name in self._declared_:
                raise AttributeError("Undeclared attribute '%s'" % name)
            self._dirty_.add(name)
            dict.__setitem__(self, name, val)
        self.__dict__[name] = val

    def __getattr__(self, name):
        if len(self._declared_) > 0 and not name in self._declared_:
            raise AttributeError("Undeclared attribute '%s'" % name)
        try:
            self._dirty_.add(name)
            return dict.__getitem__(self, name)
        except:
            if name[0] == '_':
                raise AttributeError("Unknown attribute %s" % name)
        return None

    def __repr__(self):
        return dict.__repr__(self)

    def __str__(self, nesting = 0, indent=''):
        attrs = []
        indentation = indent+"    " * nesting
        for k, v in iteritems(self.__dict__):
            if not k.startswith("_"):
                text = [indentation, k, ":"]
                if isinstance(v, CooprAPIData):
                    text.append('\n')
                    text.append(v.__str__(nesting + 1))
                else:
                    text.append(' '+str(v))
                attrs.append("".join(text))
        attrs.sort()
        return "\n".join(attrs)

    

class ICooprTask(plugin.Interface):
    """Interface for Coopr tasks"""

CooprAPIFactory = plugin.CreatePluginFactory(ICooprTask)


class TaskPlugin(plugin.Plugin, pyutilib.workflow.Task):

    def __init__(self, *args, **kwds):      #pragma:nocover
        plugin.Plugin.__init__(self, *args, **kwds)
        pyutilib.workflow.Task.__init__(self, *args, **kwds)

    def __repr__(self):
        return pyutilib.workflow.Task.__repr__(self)     #pragma:nocover


class CooprTask(TaskPlugin):

    def __init__(self, *args, **kwargs):
        self._fn = kwargs.pop('fn', None)
        #
        TaskPlugin.__init__(self, *args, **kwargs)

    def execute(self, debug=False):
        if self._fn is None:            #pragma:nocover
            raise RuntimeError("This is a bad definition of a CooprTask.  The '_fn' method is not defined")
        #
        # Process data
        #
        data = self._kwds.get('data', None)
        if not data is None and type(data) is dict:
            _data = CooprAPIData()
            _data.update(data)
            self._kwds['data'] = _data
        data = self._kwds.get('data', None)
        #
        # Test nested data
        #
        def nested_lookup(kwds, lookup):
            lookups = lookup.split('.')
            obj = kwds[lookups[0]]
            if not data is None and lookups[0] == 'data':
                data.declare(lookups[1])
            for key in lookups[1:]:
                #print key, obj
                obj = obj[key]
            if obj is None:
                raise(ValueError)
        for name in self._nested_requirements:
            try:
                nested_lookup(self._kwds, name)
            except ValueError:
                raise RuntimeError("None value found for nested attribute '%s'" % name)
            except:
                raise RuntimeError("Failed to verify existence of nested attribute '%s'" % name)
        #
        # Call _fn
        #
        if 'data' in self._kwargs:
            data = self._kwds.get('data', None)
            retval = self._fn(**self._kwds)
        else:
            data = self._kwds.pop('data')
            retval = self._fn(data, **self._kwds)
        #
        # Process retval
        #
        if retval is None or id(data) == id(retval):
            self._retval = CooprAPIData(data=data)
        elif isinstance(retval, CooprAPIData):
            if not id(data) == id(retval):
                retval.data = data
            self._retval = retval
        elif isinstance(retval, dict):
            self._retval = CooprAPIData()
            self._retval.update(retval)
        else:
            raise RuntimeError("A Coopr task function must return either None, a CooprAPIData object, or an instance of dict.")

    def _call_start(self):
        self.reset()

    def _call_init(self, *options, **kwds):
        if not 'data' in kwds:
            if len(options) > 0:
                if len(options) > 1:
                    raise RuntimeError("A CooprTask instance can only be executed with a single non-keyword argument")
                kwds['data'] = options[0]
                #options = options[1:]
            elif not self.inputs.data.optional:
                raise RuntimeError("A CooprTask instance must be executed with at 'data' argument")
        self._kwds = kwds
        return TaskPlugin._call_init(self, **kwds)

    def _call_fini(self, *options, **kwds):
        for key in self._retval:
            if not key in self.outputs:
                raise RuntimeError("Cannot return value '%s' that is not a predefined output of a Coopr task" % key)
            setattr(self, key, self._retval[key])
        TaskPlugin._call_fini(self, *options, **kwds)
        return self._retval


#
# Decorate functions that are Coopr tasks
#
def coopr_api(fn=None, implements=None, outputs=None, namespace=None):

    def my_decorator(fn):
        if fn is None:                                  #pragma:nocover
            logger.error("Error applying decorator.  No function value!")
            return

        argspec = inspect.getargspec(fn)
        if not argspec.varargs is None:
            logger.error("Attempting to declare Coopr task with function '%s' that contains variable arguments" % _alias)
            return                                      #pragma:nocover
        if not argspec.keywords is None:
            logger.error("Attempting to declare Coopr task with function '%s' that contains variable keyword arguments" % _alias)
            return                                      #pragma:nocover

        if namespace is None:
            _alias =  fn.__name__
        else:
            _alias =  namespace+'.'+fn.__name__
        _name = _alias.replace('_', '.')

        if _alias in CooprAPIFactory.services():
            logger.error("Cannot define API %s, since this API name is already defined" % _alias)
            return                                      #pragma:nocover

        class TaskMeta(plugin.PluginMeta):
            def __new__(cls, name, bases, d):
                return plugin.PluginMeta.__new__(cls, "CooprTask_"+str(_name), bases, d)


        class CooprTask_tmp(with_metaclass(TaskMeta,CooprTask)):

            plugin.alias(_alias)

            plugin.implements(ICooprTask, service=False)

            def __init__(self, *args, **kwargs):
                kwargs['fn'] = fn
                CooprTask.__init__(self, *args, **kwargs)
                if not fn is None:
                    if len(argspec.args) is 0:
                        nargs = 0
                    elif argspec.defaults is None:
                        nargs = len(argspec.args)
                    else:
                        nargs = len(argspec.args) - len(argspec.defaults)
                    self._kwargs = argspec.args[nargs:]
                    if nargs != 1 and 'data' not in self._kwargs:
                        logger.error("A Coopr functor '%s' must have a 'data argument" % _alias)
                    if argspec.defaults is None:
                        _defaults = {}
                    else:
                        _defaults = dict(list(zip(argspec.args[nargs:], argspec.defaults)))
                    #
                    docinfo = parse_docstring(fn)
                    #
                    if 'data' in docinfo['optional']:
                        self.inputs.declare('data', doc='A container of labeled data.', optional=True)
                    else:
                        self.inputs.declare('data', doc='A container of labeled data.')
                    for name in argspec.args[nargs:]:
                        if name in docinfo['optional']:
                            self.inputs.declare(name, optional=True, default=_defaults[name], doc=docinfo['optional'][name])
                        elif name in docinfo['required']:
                            self.inputs.declare(name, doc=docinfo['required'][name])
                        elif name != 'data':
                            #print docinfo
                            logger.error("Argument '%s' is not specified in the docstring!" % name)
                    #
                    self.outputs.declare('data', doc='A container of labeled data.')
                    if outputs is None:
                        _outputs = list(docinfo['return'].keys())
                    else:
                        _outputs = outputs
                    for name in _outputs:
                        if name in docinfo['return']:
                            self.outputs.declare(name, doc=docinfo['return'][name])
                        else:
                            logger.error("Return value '%s' is not specified in the docstring!" % name)
                    #
                    self._nested_requirements = []
                    for name in docinfo['required']:
                        if '.' in name:
                            self._nested_requirements.append(name)
                    #
                    #  Error check keys for docinfo
                    #
                    for name in docinfo['required']:
                        if '.' in name:
                            continue
                        if not name in self.inputs:
                            logger.error("Unexpected name '%s' in list of required inputs for functor '%s'" % (name,_alias))
                    for name in docinfo['optional']:
                        if not name in self.inputs:
                            logger.error("Unexpected name '%s' in list of optional inputs for functor '%s'" % (name,_alias))
                    for name in docinfo['return']:
                        if not name in self.outputs:
                            logger.error("Unexpected name '%s' in list of outputs for functor '%s'" % (name,_alias))
                    #
                    self.__help__ = fn.__doc__
                    self.__doc__ = fn.__doc__
                    self.__short_doc__ = docinfo['short_doc'].strip()
                    self.__long_doc__ = docinfo['long_doc'].strip()
                    self.__namespace__ = namespace

        return CooprTask_tmp()

    if fn is None:
        return my_decorator
    return my_decorator(fn)


def parse_docstring(fn):
    """Parse a function docstring for information about the function arguments and return values"""
    retval = {}
    retval['short_doc'] = ""
    retval['long_doc'] = None
    retval['required'] = {}
    retval['optional'] = {}
    retval['return'] = {}
    curr = None
    doc = inspect.getdoc(fn)
    if doc is None:
        retval['long_doc'] = ''
        return retval
    for line in doc.split('\n'):
        #print "HERE", line
        line = line.strip()
        if line == 'Required:' or line == 'Required Arguments:':
            curr = 'required'
        elif line == 'Optional:' or line == 'Optimal Arguments:':
            curr = 'optional'
        elif line == 'Return Values:' or line == 'Return:' or line == 'Returned:':
            curr = 'return'
        elif curr is None:
            if retval['long_doc'] is None:
                if line == '':
                    retval['long_doc'] = ''
                else:
                    retval['short_doc'] += line
                    retval['short_doc'] += '\n'
            else:
                retval['long_doc'] += line
                retval['long_doc'] += '\n'
        elif ':' in line:
            name, desc = line.split(':', 1)
            retval[curr][name.strip()] = desc.strip() + '\n'
        else:
            retval[curr][name.strip()] += line + '\n'
    if retval['long_doc'] is None:
        retval['long_doc'] = ''
    return retval

