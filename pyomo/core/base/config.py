import appdirs
import os
import json
try:
    import yaml
    yaml_available = True
except ImportError:
    yaml_available = False

from pyutilib.misc.config import ConfigBase, ConfigBlock, ConfigValue

USER_OPTION = 0
ADVANCED_OPTION = 1
DEVELOPER_OPTION = 2


class PyomoOptions_(object):

    def __init__(self):
        self._options_stack = [ default_pyomo_config() ]

        # Load the user's configuration
        sources = [(json.load, 'json')]
        if yaml_available:
            sources.append( (yaml.load, 'yml') )
            sources.append( (yaml.load, 'yaml') )
        for parser, suffix in sources:
            cfg_file = os.path.join( appdirs.user_data_dir('pyomo'),
                                     'config.'+suffix)
            if os.path.exists(cfg_file):
                fp = open(cfg_file)
                try:
                    data = parser(fp)
                except:
                    logger.error("Error parsing the user's default "
                                 "configuration file\n\t%s." % (cfg_file,))
                self._options_stack[0].set_value(data)

    def active_config(self):
        return self._options_stack[-1]

    #
    # BEGIN clone of the ConfigBlock API
    #

    def __getitem__(self, key):
        return self.active_config().__getitem__(key)

    def get(self, key, default=ConfigBase.NoArgument):
        return self.active_config().get(key,default)

    def __setitem__(self, key, val):
        return self.active_config().__setitem__(key, val)

    def __contains__(self, key):
        return self.active_config().__contains__(key)

    def __len__(self):
        return self.active_config().__len__()

    def __iter__(self):
        return self.active_config().__iter__()

    def __getattr__(self, name):
        #if name in self.__dict__:
        #    return self.__dict__[name]
        return self._options_stack[-1].__getattr__(name)

    def __setattr__(self, name, value):
        if name == '_options_stack':
            super(PyomoOptions_,self).__setattr__(name, value)
        else:
            return self.active_config().__setattr__(name, value)

    def iterkeys(self):
        return self.active_config().iterkeys()

    def itervalues(self):
        return self.active_config().itervalues()

    def iteritems(self):
        return self.active_config().iteritems()

    #
    # Note: ConfigBlock correctly maps {keys,values,items} to
    # iter{keys,values,items} when running under Python3 ... so we
    # *don't* need to do that mapping here.
    #
    def keys(self):
        return self.active_config().keys()

    def values(self):
        return self.active_config().values()

    def items(self):
        return self.active_config().items()

    def declare(self, name, config):
        return self.active_config().items(name, config)

    def add(self, name, config):
        return self.active_config().add(name, value)

    def value(self, accessValue=True):
        return self.active_config().value(accessValue)

    def set_value(self, value):
        return self.active_config().set_value(value)

    def reset(self):
        return self.active_config().reset()

    #
    # END clone of the ConfigBlock API
    #



def default_pyomo_config():
    config = ConfigBlock("Pyomo configuration file")

    config.declare('paranoia_level', ConfigValue(
        0, int,
        'Pyomo paranoia and error checking level',
        """Higher levels of paranoia enable additional error checking and
        warning messages that may assist users in identifying likely
        modeling problems.
        Default=0""",
        visibility=ADVANCED_OPTION ) )

    return config


PyomoOptions = PyomoOptions_()
