#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
#  This module was originally developed as part of the PyUtilib project
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  ___________________________________________________________________________

import os
import time
import tempfile
import logging
import shutil
import weakref
from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.common.errors import TempfileContextError
from pyomo.common.multithread import MultiThreadWrapperWithMain

try:
    from pyutilib.component.config.tempfiles import TempfileManager as pyutilib_mngr
except ImportError:
    pyutilib_mngr = None

deletion_errors_are_fatal = True

logger = logging.getLogger(__name__)


class TempfileManagerClass(object):
    """A class for managing tempfile contexts

    Pyomo declares a global instance of this class as ``TempfileManager``:

    .. doctest::

       >>> from pyomo.common.tempfiles import TempfileManager

    This class provides an interface for managing
    :class:`TempfileContext` contexts.  It implements a basic stack,
    where users can :meth:`push()` a new context (causing it to become
    the current "active" context) and :meth:`pop()` contexts off
    (optionally deleting all files associated with the context).  In
    general usage, users will either use this class to create new
    tempfile contexts and use them explicitly (i.e., through a context
    manager):

    .. doctest::

       >>> import os
       >>> with TempfileManager.new_context() as tempfile:
       ...     fd, fname = tempfile.mkstemp()
       ...     dname = tempfile.mkdtemp()
       ...     os.path.isfile(fname)
       ...     os.path.isdir(dname)
       True
       True
       >>> os.path.exists(fname)
       False
       >>> os.path.exists(dname)
       False

    or through an implicit active context accessed through the manager
    class:

    .. doctest::

       >>> TempfileManager.push()
       <pyomo.common.tempfiles.TempfileContext object ...>
       >>> fname = TempfileManager.create_tempfile()
       >>> dname = TempfileManager.create_tempdir()
       >>> os.path.isfile(fname)
       True
       >>> os.path.isdir(dname)
       True

       >>> TempfileManager.pop()
       <pyomo.common.tempfiles.TempfileContext object ...>
       >>> os.path.exists(fname)
       False
       >>> os.path.exists(dname)
       False

    """

    def __init__(self):
        self._context_stack = []
        self._context_manager_stack = []
        self.tempdir = None

    def __del__(self):
        self.shutdown()

    def shutdown(self, remove=True):
        if not self._context_stack:
            return
        if any(ctx.tempfiles for ctx in self._context_stack):
            logger.error(
                "Temporary files created through TempfileManager "
                "contexts have not been deleted (observed during "
                "TempfileManager instance shutdown).\n"
                "Undeleted entries:\n\t"
                + "\n\t".join(
                    fname if isinstance(fname, str) else fname.decode()
                    for ctx in self._context_stack
                    for fd, fname in ctx.tempfiles
                )
            )
        if self._context_stack:
            logger.warning(
                "TempfileManagerClass instance: un-popped tempfile "
                "contexts still exist during TempfileManager instance "
                "shutdown"
            )
        self.clear_tempfiles(remove)
        # Delete the stack so that subsequent operations generate an
        # exception
        self._context_stack = None

    def context(self):
        """Return the current active TempfileContext.

        Raises
        ------
        TempfileContextError if there is not a current context."""
        if not self._context_stack:
            raise TempfileContextError(
                "TempfileManager has no currently active context.  "
                "Create a context (with push() or __enter__()) before "
                "attempting to create temporary objects."
            )
        return self._context_stack[-1]

    def create_tempfile(self, suffix=None, prefix=None, text=False, dir=None):
        "Call :meth:`TempfileContext.create_tempfile` on the active context"
        return self.context().create_tempfile(
            suffix=suffix, prefix=prefix, text=text, dir=dir
        )

    def create_tempdir(self, suffix=None, prefix=None, dir=None):
        "Call :meth:`TempfileContext.create_tempdir` on the active context"
        return self.context().create_tempdir(suffix=suffix, prefix=prefix, dir=dir)

    def add_tempfile(self, filename, exists=True):
        "Call :meth:`TempfileContext.add_tempfile` on the active context"
        return self.context().add_tempfile(filename=filename, exists=exists)

    def clear_tempfiles(self, remove=True):
        """Delete all temporary files and remove all contexts."""
        while self._context_stack:
            self.pop(remove)

    @deprecated(
        "The TempfileManager.sequential_files() method has been "
        "removed.  All temporary files are created with guaranteed "
        "unique names.  Users wishing sequentially numbered files "
        "should create a temporary (empty) directory using mkdtemp "
        "/ create_tempdir and place the sequential files within it.",
        version='6.2',
    )
    def sequential_files(self, ctr=0):
        pass

    def unique_files(self):
        pass

    def new_context(self):
        """Create and return an new tempfile context

        Returns
        -------
        TempfileContext
            the newly-created tempfile context

        """
        return TempfileContext(self)

    def push(self):
        """Create a new tempfile context and set it as the active context.

        Returns
        -------
        TempfileContext
            the newly-created tempfile context

        """
        context = self.new_context()
        self._context_stack.append(context)
        return context

    def pop(self, remove=True):
        """Remove and release the active context

        Parameters
        ----------
        remove: bool
            If ``True``, delete all managed files / directories

        """
        ctx = self._context_stack.pop()
        ctx.release(remove)
        return ctx

    def __enter__(self):
        ctx = self.push()
        self._context_manager_stack.append(ctx)
        return ctx

    def __exit__(self, exc_type, exc_val, exc_tb):
        ctx = self._context_manager_stack.pop()
        while True:
            if ctx is self.pop():
                break
            logger.warning(
                "TempfileManager: tempfile context was pushed onto "
                "the TempfileManager stack within a context manager "
                "(i.e., `with TempfileManager:`) but was not popped "
                "before the context manager exited.  Popping the "
                "context to preserve the stack integrity."
            )


class TempfileContext:
    """A `context` for managing collections of temporary files

    Instances of this class hold a "temporary file context".  That is,
    this records a collection of temporary file system objects that are
    all managed as a group.  The most common use of the context is to
    ensure that all files are deleted when the context is released.

    This class replicates a significant portion of the :mod:`tempfile`
    module interface.

    Instances of this class may be used as context managers (with the
    temporary files / directories getting automatically deleted when the
    context manager exits).

    Instances will also attempt to delete any temporary objects from the
    filesystem when the context falls out of scope (although this
    behavior is not guaranteed for instances existing when the
    interpreter is shutting down).

    """

    def __init__(self, manager):
        self.manager = weakref.ref(manager)
        self.tempfiles = []
        self.tempdir = None

    def __del__(self):
        self.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def mkstemp(self, suffix=None, prefix=None, dir=None, text=False):
        """Create a unique temporary file using :func:`tempfile.mkstemp`

        Parameters are handled as in :func:`tempfile.mkstemp`, with
        the exception that the new file is created in the directory
        returned by :meth:`gettempdir`

        Returns
        -------
        fd: int
            the opened file descriptor

        fname: str or bytes
            the absolute path to the new temporary file

        """
        dir = self._resolve_tempdir(dir)
        # Note: ans == (fd, fname)
        ans = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir, text=text)
        self.tempfiles.append(ans)
        return ans

    def mkdtemp(self, suffix=None, prefix=None, dir=None):
        """Create a unique temporary directory using :func:`tempfile.mkdtemp`

        Parameters are handled as in :func:`tempfile.mkdtemp`, with
        the exception that the new file is created in the directory
        returned by :meth:`gettempdir`

        Returns
        -------
        dname: str or bytes
            the absolute path to the new temporary directory

        """
        dir = self._resolve_tempdir(dir)
        dname = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
        self.tempfiles.append((None, dname))
        return dname

    def gettempdir(self):
        """Return the default name of the directory used for temporary files.

        This method returns the first non-null location returned from:

         - This context's ``tempdir`` (i.e., ``self.tempdir``)
         - This context's manager's ``tempdir`` (i.e.,
           ``self.manager().tempdir``)
         - :func:`tempfile.gettempdir()`

        Returns
        -------
        dir: str
            The default directory to use for creating temporary objects
        """
        dir = self._resolve_tempdir()
        if dir is None:
            return tempfile.gettempdir()
        if isinstance(dir, bytes):
            return dir.decode()
        return dir

    def gettempdirb(self):
        """Same as :meth:`gettempdir()`, but the return value is ``bytes``"""
        dir = self._resolve_tempdir()
        if dir is None:
            return tempfile.gettempdirb()
        if not isinstance(dir, bytes):
            return dir.encode()
        return dir

    def gettempprefix(self):
        """Return the filename prefix used to create temporary files.

        See :func:`tempfile.gettempprefix()`

        """
        return tempfile.gettempprefix()

    def gettempprefixb(self):
        """Same as :meth:`gettempprefix()`, but the return value is ``bytes``"""
        return tempfile.gettempprefixb()

    def create_tempfile(self, suffix=None, prefix=None, text=False, dir=None):
        """Create a unique temporary file.

        The file name is generated as in :func:`tempfile.mkstemp()`.

        Any file handles to the new file (e.g., from :meth:`mkstemp`)
        are closed.

        Returns
        -------
        fname: str or bytes
            The absolute path of the new file.

        """
        fd, fname = self.mkstemp(suffix=suffix, prefix=prefix, dir=dir, text=text)
        os.close(fd)
        self.tempfiles[-1] = (None, fname)
        return fname

    def create_tempdir(self, suffix=None, prefix=None, dir=None):
        """Create a unique temporary directory.

        The file name is generated as in :func:`tempfile.mkdtemp()`.

        Returns
        -------
        dname: str or bytes
            The absolute path of the new directory.

        """
        return self.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)

    def add_tempfile(self, filename, exists=True):
        """Declare the specified file/directory to be temporary.

        This adds the specified path as a "temporary" object to this
        context's list of managed temporary paths (i.e., it will be
        potentially be deleted when the context is released (see
        :meth:`release`).

        Parameters
        ----------
        filename: str
            the file / directory name to be treated as temporary
        exists: bool
            if ``True``, the file / directory must already exist.

        """
        tmp = os.path.abspath(filename)
        if exists and not os.path.exists(tmp):
            raise IOError("Temporary file does not exist: " + tmp)
        self.tempfiles.append((None, tmp))

    def release(self, remove=True):
        """Release this context

        This releases the current context, potentially deleting all
        managed temporary objects (files and directories), and resetting
        the context to generate unique names.

        Parameters
        ----------
        remove: bool
            If ``True``, delete all managed files / directories
        """
        if remove:
            for fd, name in reversed(self.tempfiles):
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                self._remove_filesystem_object(name)
        self.tempfiles.clear()

    def _resolve_tempdir(self, dir=None):
        if dir is not None:
            return dir
        elif self.tempdir is not None:
            return self.tempdir
        elif self.manager().tempdir is not None:
            return self.manager().tempdir
        elif TempfileManager.main_thread.tempdir is not None:
            return TempfileManager.main_thread.tempdir
        elif pyutilib_mngr is not None and pyutilib_mngr.tempdir is not None:
            deprecation_warning(
                "The use of the PyUtilib TempfileManager.tempdir "
                "to specify the default location for Pyomo "
                "temporary files has been deprecated.  "
                "Please set TempfileManager.tempdir in "
                "pyomo.common.tempfiles",
                version='5.7.2',
            )
            return pyutilib_mngr.tempdir
        return None

    def _remove_filesystem_object(self, name):
        if not os.path.exists(name):
            return
        if os.path.isfile(name) or os.path.islink(name):
            try:
                os.remove(name)
            except WindowsError:
                # Sometimes Windows doesn't release the
                # file lock immediately when the process
                # terminates.  If we get an error, wait a
                # second and try again.
                try:
                    time.sleep(1)
                    os.remove(name)
                except WindowsError:
                    if deletion_errors_are_fatal:
                        raise
                    else:
                        # Failure to delete a tempfile
                        # should NOT be fatal
                        logger = logging.getLogger(__name__)
                        logger.warning("Unable to delete temporary file %s" % (name,))
            return
        assert os.path.isdir(name)
        shutil.rmtree(name, ignore_errors=not deletion_errors_are_fatal)


# The global Pyomo TempfileManager instance
TempfileManager: TempfileManagerClass = MultiThreadWrapperWithMain(TempfileManagerClass)
