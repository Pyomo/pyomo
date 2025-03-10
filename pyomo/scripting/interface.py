#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.plugin_base import (
    Interface,
    DeprecatedInterface,
    Plugin,
    SingletonPlugin,
    ExtensionPoint,
    implements,
    alias,
)

registered_callback = {}


def pyomo_callback(name):
    """This is a decorator that declares a function to be
    a callback function.  The callback functions are
    added to the solver when run from the pyomo script.

    Example:

    .. code::

        @pyomo_callback('cut-callback')
        def my_cut_generator(solver, model):
            ...

    """

    def fn(f):
        registered_callback[name] = f
        return f

    return fn


class IPyomoScriptPreprocess(Interface):
    def apply(self, **kwds):
        """Apply preprocessing step in the Pyomo script"""


class IPyomoScriptCreateModel(Interface):
    def apply(self, **kwds):
        """Apply model creation step in the Pyomo script"""


class IPyomoScriptModifyInstance(Interface):
    def apply(self, **kwds):
        """Modify and return the model instance"""


class IPyomoScriptCreateDataPortal(Interface):
    def apply(self, **kwds):
        """Apply model data creation step in the Pyomo script"""


class IPyomoScriptPrintModel(Interface):
    def apply(self, **kwds):
        """Apply model printing step in the Pyomo script"""


class IPyomoScriptPrintInstance(Interface):
    def apply(self, **kwds):
        """Apply instance printing step in the Pyomo script"""


class IPyomoScriptSaveInstance(Interface):
    def apply(self, **kwds):
        """Apply instance saving step in the Pyomo script"""


class IPyomoScriptPrintResults(Interface):
    def apply(self, **kwds):
        """Apply results printing step in the Pyomo script"""


class IPyomoScriptSaveResults(Interface):
    def apply(self, **kwds):
        """Apply results saving step in the Pyomo script"""


class IPyomoScriptPostprocess(Interface):
    def apply(self, **kwds):
        """Apply postprocessing step in the Pyomo script"""


class IPyomoPresolver(Interface):
    def get_actions(self):
        """Return a list of presolve actions, in the order in which
        they will be applied."""

    def activate_action(self, action):
        """Activate an action, but leave its default rank"""

    def deactivate_action(self, action):
        """Deactivate an action"""

    def set_actions(self, actions):
        """Set presolve action list"""

    def presolve(self, instance):
        """Apply the presolve actions to this instance, and return the
        revised instance"""


class IPyomoPresolveAction(Interface):
    def presolve(self, instance):
        """Apply the presolve action to this instance, and return the
        revised instance"""

    def rank(self):
        """Return an integer that is used to automatically order presolve actions,
        from low to high rank."""
