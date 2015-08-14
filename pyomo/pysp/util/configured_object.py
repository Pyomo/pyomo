import inspect
import copy
import sys

from pyutilib.misc.config import (ConfigValue,
                                  ConfigBlock)
from pyomo.pysp.util.config import (safe_register_option,
                                    check_options_match)
import six

class PySPConfiguredObject(object):

    _registered_options = \
        ConfigBlock("Options registered for the PySPConfiguredObject class")

    def __init__(self, options):
        self._options = None
        self.set_options(options)

    def set_options(self, options):
        self.validate_options(options)
        self._options = options

    def display_options(self, ostream=sys.stdout):
        max_len = max(len(name) for name in self._options)
        line_template = " %s %"+str(max_len)+"s: %s\n"
        for name in self._options:
            configval = self._options.get(name)
            flag = "-"
            if configval._userSet:
                flag = "*"
            ostream.write(line_template
                          % (flag, name,configval.value(accessValue=False)))

    @classmethod
    def register_options(cls, options=None, include_bases=True):
        if options is None:
            options = ConfigBlock()
        if include_bases:
            bases = inspect.getmro(cls)
            assert bases[-1] is object
        else:
            bases = (cls,)
        for cls in reversed(bases):
            if issubclass(cls, PySPConfiguredObject):
                for name in cls._registered_options:
                    configval = cls._registered_options.get(name)
                    assert configval._parent is cls._registered_options
                    configval._parent = None
                    safe_register_option(options, name, configval)
                    configval._parent = cls._registered_options
        return options

    @classmethod
    def extract_options(cls,
                        options,
                        error_if_missing=True,
                        include_bases=True):

        these_options = ConfigBlock()
        cls.register_options(these_options, include_bases=include_bases)
        for name in these_options.keys():
            configval = None
            try:
                configval = options.get(name)
            except KeyError:
                configval = None
                if error_if_missing:
                    raise
                else:
                    continue
            assert configval is not None
            this_configval = these_options.get(name)
            check_options_match(this_configval,
                                configval,
                                include_value=False,
                                include_accessed=False)
            if configval._userSet:
                this_configval.set_value(configval.value())

        return these_options

    @classmethod
    def extract_user_options_to_dict(cls,
                                     options,
                                     error_if_missing=True,
                                     include_bases=True):
        these_options = \
            cls.extract_options(options,
                                error_if_missing=error_if_missing,
                                include_bases=include_bases)
        options_dict = {}
        for name in these_options:
            configval = these_options.get(name)
            if configval._userSet:
                options_dict[name] = configval.value()
        return options_dict

    @classmethod
    def validate_options(cls,
                         options,
                         error_if_missing=True,
                         include_bases=True):

        if include_bases:
            bases = inspect.getmro(cls)
            assert bases[-1] is object
        else:
            bases = (cls,)
        for cls in reversed(bases):
            if issubclass(cls, PySPConfiguredObject):
                for name in cls._registered_options:
                    configval = None
                    try:
                        configval = options.get(name)
                    except KeyError:
                        configval = None
                        if error_if_missing:
                            raise
                        else:
                            continue
                    assert configval is not None
                    this_configval = cls._registered_options.get(name)
                    check_options_match(this_configval,
                                        configval,
                                        include_value=False,
                                        include_accessed=False)
