#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import ctypes
import os
import six

from six import iteritems

class _MsvcrtInterface(object):
    """Helper class to amnage the interface with the MSVCRT runtime"""

    def __init__(self):
        self._loaded = None
        self.putenv_s = _MsvcrtInterface.noop_put
        self.wputenv_s = _MsvcrtInterface.noop_put
        self.getenv = _MsvcrtInterface.noop_get
        self.wgetenv = _MsvcrtInterface.noop_get

    @staticmethod
    def noop_get(key):
        return None

    @staticmethod
    def noop_put(key, val):
        return 0

    def available(self):
        if self._loaded is not None:
            return self._loaded

        try:
            msvcrt = ctypes.cdll.msvcrt
            self._loaded = True
        except OSError:
            self._loaded = False
            return self._loaded

        self.putenv_s = ctypes.cdll.msvcrt._putenv_s
        self.putenv_s.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.putenv_s.restype = ctypes.c_int

        self.wputenv_s = ctypes.cdll.msvcrt._wputenv_s
        self.wputenv_s.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        self.wputenv_s.restype = ctypes.c_int

        self.getenv = ctypes.cdll.msvcrt.getenv
        self.getenv.argtypes = [ctypes.c_char_p]
        self.getenv.restype = ctypes.c_char_p

        self.wgetenv = ctypes.cdll.msvcrt._wgetenv
        self.wgetenv.argtypes = [ctypes.c_wchar_p]
        self.wgetenv.restype = ctypes.c_wchar_p

        return self._loaded

def _as_bytes(val):
    """Helper function to coerce a string to a bytes() object"""
    if isinstance(val, six.binary_type):
        return val
    else:
        return val.encode('utf-8')

def _as_unicode(val):
    """Helper function to coerce a string to a unicode() object"""
    if isinstance(val, six.text_type):
        return val
    else:
        return val.decode()

class TemporaryEnv(object):
    msvcrt = _MsvcrtInterface()

    def __init__(self, **kwds):
        """A context manager for managing environment variables

        This class provides a simplified interface for consistently
        setting and restoring environment variables, with special
        handling to ensure consistency on Windows platforms.

        `os.environ` reflects the current python environment variables,
        and will be passed to subprocesses.  However, it does not
        reflect the MSVCRT environment on Windows platforms.  This can
        be problemmatic as DLLs loaded through the CTYPES interface
        will see the MSVCRT environment and not os.environ.  This class
        provides a way to manage environment variables and pass changes
        to both os.environ and the MSVCRT runtime.

        This class implements a context manager API, so that clients can
        temporarily change - and then subsequently restore - the
        environment.

        .. testcode::
           :hide:

           import os
           from pyomo.common.env import TemporaryEnv
           orig_env_val = os.environ.get('TEMP_ENV_VAR', None)

        .. doctest::

           >>> os.environ['TEMP_ENV_VAR'] = 'original value'
           >>> print(os.environ['TEMP_ENV_VAR'])
           original value

           >>> with TemporaryEnv(TEMP_ENV_VAR='temporary value'):
           ...    print(os.envion['TEMP_ENV_VAR'])
           temporary value

           >>> print(os.environ['TEMP_ENV_VAR'])
           original value

        .. testcode::
           :hide:

           if orig_env_val is None:
               del os.environ['TEMP_ENV_VAR']
           else:
               os.environ['TEMP_ENV_VAR'] = orig_env_val

        """
        self.original_state = {}
        for k, v in iteritems(kwds):
            self[k] = v

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.restore()

    def restore(self):
        """Restore the environment to the original state

        This restores all environment variables modified through this
        object to the state they were in before this instance made any
        changes.  Note that any changes made directly to os.environ
        outside this instance will not be detected / undone.

        """
        # It is possible that os.environ and the MSVCRT did not start
        # off in sync; e.g., if someone had set a value in os.environ
        # directly.  We will be especially careful and restore the
        # twoenvironments to their original state (even if it was not
        # synchronized)

        # Restore os.environ
        for key, val in iteritems(self.original_state):
            if val[0] is None:
                try:
                    del os.environ[key]
                except KeyError:
                    pass
            else:
                os.environ[key] = val[0]

        if self.msvcrt.available():
            # Restore MSVCRT
            for key, val in iteritems(self.original_state):
                if val[1] is None:
                    if isinstance(key, six.text_type):
                        self.msvcrt.wputenv_s(key, u'')
                    else:
                        self.msvcrt.putenv_s(key, b'')
                elif isinstance(val[1], six.text_type):
                    self.msvcrt.wputenv_s(_as_unicode(key), val[1])
                else:
                    self.msvcrt.putenv_s(_as_bytes(key), val[1])

        self.original_state = {}

    def __getitem__(self, key):
        """Retun the current environment variable value from os.environ"""
        return os.environ[key]

    def __contains__(self, key):
        """Retun true if the key is in os.environ"""
        return key in os.environ

    def __setitem__(self, key, val):
        if not self.msvcrt.available():
            _msvcrt_orig = None
        else:
            # interfacing with the MSVCRT can be tricky, especially
            # under Python 3.x: it is critical that all arguments
            # "match"; that is, either everything is Unicode through the
            # "w" interfaces or everything is a char* through the
            # "non-w" interface.
            if isinstance(key, six.text_type) or isinstance(val, six.text_type):
                _key = _as_unicode(key)
                _val = _as_unicode(val)
                _msvcrt_orig = self.msvcrt.wgetenv(_key)
                self.msvcrt.wputenv_s(_key, _val)
            else:
                _key = _as_bytes(key)
                _val = _as_bytes(val)
                _msvcrt_orig = self.msvcrt.getenv(_key)
                self.msvcrt.putenv_s(_key, _val)

        if key not in self.original_state:
            self.original_state[key] = (
                os.environ.get(key, None), _msvcrt_orig
            )

        os.environ[key] = val

    def __delitem__(self, key):
        if key not in self.original_state:
            # cause the original state to be recorded
            self[key] = ''

        try:
            del os.environ[key]
        except KeyError:
            return

        if self.msvcrt.available():
            if isinstance(key, six.text_type):
                self.msvcrt.wputenv_s(key, u'')
            else:
                self.msvcrt.putenv_s(key, b'')

