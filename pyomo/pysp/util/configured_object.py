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
        """Assign an options block to this class instance after
        validating that all registered options for this class exist on
        the block."""
        self.validate_options(options)
        self._options = options

    def display_options(self, ostream=sys.stdout):
        """Display the options block currently assigned to this class
        instance."""
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
    def register_options(cls, *args):
        """Cls.register_options([options]) -> options.
        Fills an options block will all registered options for this
        class. The optional argument 'options' can be a previously
        existing options block, which would be both updated and
        returned by this function."""

        if len(args) == 0:
            options = ConfigBlock()
        else:
            if len(args) != 1:
                raise TypeError("register_options(...) takes at most 1 argument (%s given)"
                                % (len(args)))
            options = args[0]
            if not isinstance(options, ConfigBlock):
                raise TypeError("register_options(...) argument must be of type ConfigBlock, "
                                "not %s" % (type(options).__name__))

        bases = inspect.getmro(cls)
        assert bases[-1] is object
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
    def extract_options(cls, options, error_if_missing=True):
        """Copy the set of registered options for this class from an
        existing options block and return a new options block with
        only those values. This method will preserve the _userSet
        status of all options. The optional flag 'error_if_missing'
        controls whether or not an exception is raised when registered
        options are missing."""

        these_options = cls.register_options()
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
                                     sparse=False):
        """Copy the set of registered options for this class from an
        existing options block and return a dictionary of options
        (name -> value) with those values. This method will preserve
        the _userSet status of all options. The optional flag
        'error_if_missing' controls whether or not an exception is
        raised when registered options are missing. The optional flag
        'sparse' controls whether non user-set values should be
        included in the returned dictionary."""

        these_options = \
            cls.extract_options(options,
                                error_if_missing=error_if_missing)
        options_dict = {}
        for name in these_options:
            configval = these_options.get(name)
            if (not sparse) or configval._userSet:
                options_dict[name] = configval.value()
        return options_dict

    @classmethod
    def validate_options(cls, options, error_if_missing=True):
        """Validate that all registered can be found in the options
        block and that their option definitions are the same. The
        options flag 'error_if_missing' can be used to control whether
        or not an exception is raised when registered options are
        missing."""

        bases = inspect.getmro(cls)
        assert bases[-1] is object
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
