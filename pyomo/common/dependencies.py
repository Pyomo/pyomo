#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import importlib
import logging

class DeferredImportError(ImportError):
    pass

class ModuleUnavailable(object):
    """Dummy object that raises a DeferredImportError upon attribute access

    This object is returned by attempt_import() in liu of the module in
    the case that the module import fails.  Any attempts to access
    attributes on this object will raise a DeferredImportError
    exception.

    Parameters
    ----------
    message: str
        The string message to return in the raised exception
    """
    def __init__(self, message):
        self._error_message_ = message

    def __getattr__(self, attr):
        raise DeferredImportError(self._error_message_)

    def generate_import_warning(self, logger='pyomo.common'):
        logging.getLogger(logger).warning(
            self._error_message_)

try:
    from packaging import version as _version
    _parser = _version.parse
except ImportError:
    # pkg_resources is an order of magnitude slower to import than
    # packaging.  Only use it if the preferred (but optional) packaging
    # library is not present
    from pkg_resources import parse_version as _parser

def _check_version(module, min_version):
    return _parser(min_version) <= _parser(module.__version__)
    

def attempt_import(name, error_message=None, only_catch_importerror=True,
                   minimum_version=None):
    """Attempt to import the specified module.

    This will attempt to import the specified module, returning a
    (module, available) tuple.  If the import was successful, `module`
    will be the imported module and `available` will be True.  If the
    import results in an exception, then `module` will be an instance of
    :py:class:`ModuleUnavailable` and `available` will be False

    The following is equivalent to ``import numpy as np``:

    .. doctest::

       >>> from pyomo.common.dependencies import attempt_import
       >>> np, numpy_available = attempt_import('numpy')

    Parameters
    ----------
    name: `str`
        The name of the module to import

    error_message: `str`, optional
        The message for the exception raised by ModuleUnavailable

    only_catch_importerror: `bool`, optional
        If True, exceptions other than ImportError raised during module
        import will be reraised.  If False, any exception will result in
        returning a ModuleUnavailable object.

    Returns
    -------
    : module
        the imported module or an instance of :py:class:`ModuleUnavailable`
    : bool
        Boolean indicating if the module import succeeded
    """
    try:
        module = importlib.import_module(name)
        if minimum_version is None:
            return module, True
        elif _check_version(module, minimum_version):
            return module, True
        elif error_message:
            error_message += " (version %s does not satisfy the minimum " \
                             "version %s)" % (
                                 module.__version__, minimum_version)
        else:
            error_message = "The %s module version %s does not satisfy " \
                            "the minimum version %s" % (
                                name, module.__version__.minimum_version)
    except ImportError:
        pass
    except:
        if only_catch_importerror:
            raise

    if not error_message:
        error_message = "The %s module (an optional Pyomo dependency) " \
                        "failed to import" % (name,)
    return ModuleUnavailable(error_message), False

#
# Common optional dependencies used throughout Pyomo
#

yaml, yaml_available = attempt_import('yaml')
if yaml_available and hasattr(yaml, 'SafeLoader'):
    yaml_load_args = {'Loader': yaml.SafeLoader}
else:
    yaml_load_args = {}

numpy, numpy_available = attempt_import('numpy')
scipy, scipy_available = attempt_import('scipy')
