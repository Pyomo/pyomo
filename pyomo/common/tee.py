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
#
#  This module was originally developed as part of the PyUtilib project
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  ___________________________________________________________________________
#
import collections.abc
import io
import logging
import os
import sys
import threading
import time

from pyomo.common.log import LoggingIntercept, LogStream

_poll_interval = 0.0001
_poll_rampup_limit = 0.099
# reader polling: number of timeouts with no data before increasing the
# polling interval
_poll_rampup = 10
# polling timeout when waiting to close threads.  This will bail on
# closing threast after a minimum of 13.1 seconds and a worst case of
# ~(13.1 * #threads) seconds
_poll_timeout = 3  # 15 rounds: 0.0001 * 2**15 == 3.2768
_poll_timeout_deadlock = 200  # 21 rounds: 0.0001 * 2**21 == 209.7152

_mswindows = sys.platform.startswith('win')
try:
    if _mswindows:
        from msvcrt import get_osfhandle
        from win32pipe import PeekNamedPipe
        from win32file import ReadFile
    else:
        from select import select
    _peek_available = True
except ImportError:
    _peek_available = False

logger = logging.getLogger(__name__)


class _SignalFlush(object):
    def __init__(self, ostream, handle):
        super().__setattr__('_ostream', ostream)
        super().__setattr__('_handle', handle)

    def flush(self):
        self._ostream.flush()
        self._handle.flush = True

    def __getattr__(self, attr):
        return getattr(self._ostream, attr)

    def __setattr__(self, attr, val):
        return setattr(self._ostream, attr, val)


class _AutoFlush(_SignalFlush):
    def write(self, data):
        self._ostream.write(data)
        self.flush()

    def writelines(self, data):
        self._ostream.writelines(data)
        self.flush()


class redirect_fd(object):
    """Redirect a file descriptor to a new file or file descriptor.

    This context manager will redirect the specified file descriptor to
    a specified new output target (either file name or file descriptor).
    For the special case of file descriptors 1 (stdout) and 2 (stderr),
    we will also make sure that the Python `sys.stdout` or `sys.stderr`
    remain usable: in the case of synchronize=True, the `sys.stdout` /
    `sys.stderr` file handles point to the new file descriptor.  When
    synchronize=False, we preserve the behavior of the Python file
    object (retargeting it to the original file descriptor if necessary).

    Parameters
    ----------
    fd: int
        The file descriptor to redirect

    output: int or str or None
        The new output target for `fd`: either another valid file
        descriptor (int) or a string with the file to open.  If `None`,
        then the fd is redirected to `os.devnull`.

    synchronize: bool
        If True, and `fd` is 1 or 2, then update `sys.stdout` or
        `sys.stderr` to also point to the new file descriptor

    """

    def __init__(self, fd=1, output=None, synchronize=True):
        if output is None:
            # /dev/null is used just to discard what is being printed
            output = os.devnull
        self.fd = fd
        self.std = {1: 'stdout', 2: 'stderr'}.get(self.fd, None)
        self.target = output
        self.target_file = None
        self.synchronize = synchronize
        self.original_file = None
        self.original_fd = None

    def __enter__(self):
        if self.std:
            self.original_file = getattr(sys, self.std)
            # important: flush the current file buffer when redirecting
            self.original_file.flush()
        # Duplicate the original standard file descriptor(file
        # descriptor 1 or 2) to a different file descriptor number
        self.original_fd = os.dup(self.fd)

        # Open a file descriptor pointing to the new file
        if isinstance(self.target, int):
            out_fd = self.target
        else:
            out_fd = os.open(self.target, os.O_WRONLY)

        # Duplicate the file descriptor for the opened file, closing and
        # overwriting/replacing the original fd.  Only make the new FD
        # inheritable if it is stdout/stderr
        os.dup2(out_fd, self.fd, inheritable=bool(self.std))

        # We no longer need this original file descriptor
        if not isinstance(self.target, int):
            os.close(out_fd)

        if self.synchronize and self.std:
            # Cause Python's stdout to point to our new file
            self.target_file = os.fdopen(self.fd, 'w', closefd=False)
            setattr(sys, self.std, self.target_file)

        return self

    def __exit__(self, et, ev, tb):
        # Close output: this either closes the new file that we opened,
        # or else the new file that points to the original (duplicated)
        # file descriptor
        if self.target_file is not None:
            self.target_file.flush()
            self.target_file.close()
            self.target_file = None
            setattr(sys, self.std, self.original_file)
        # Restore stdout's FD (implicitly closing the FD we opened)
        os.dup2(self.original_fd, self.fd, inheritable=bool(self.std))
        # Close the temporary FD
        os.close(self.original_fd)


class capture_output(object):
    """Context manager to capture output sent to sys.stdout and sys.stderr

    This is a drop-in substitute for PyUtilib's capture_output to
    temporarily redirect output to the provided stream or file.

    Parameters
    ----------
    output : io.TextIOBase, Sequence[io.TextIOBase], TeeStream, str, or None

        Output stream where all captured stdout/stderr data is sent.  If
        a ``str`` is provided, it is used as a file name and opened
        (potentially overwriting any existing file).  If ``None``, a
        :class:`io.StringIO` object is created and used.

    capture_fd : bool

        If True, we will also redirect the process file descriptors
        ``1`` (stdout), ``2`` (stderr), and the file descriptors from
        ``sys.stdout.fileno()`` and ``sys.stderr.fileno()`` to the
        ``output``.  This is useful for capturing output emitted
        directly to the process stdout / stderr by external compiled
        modules.

        Capturing and redirecting the file descriptors can cause loops
        in the output stream (where one of the `output` streams points
        to a file descriptor we just captured).
        :py:class:`capture_output` will attempt to locate
        :py:class:`io.IOBase` streams in `output` that point to file
        descriptors that we just captured and replace them with
        temporary streams that point to (copies of) the original file
        descriptor.  In addition, :py:class:`capture_output` will look
        for :py:class:`~pyomo.common.log.LogStream` objects and will
        attempt to locate :py:class:`logging.StreamHandle` objects that
        would output to a redirected file descriptor and temporarily
        redirect those handlers to (copies of) the original file
        descriptor.

        Note that this process will cover the most common cases, but is
        by no means perfect.  Use of other output classes or customized
        log handlers may still result in output loops (usually
        manifesting in an error message about text being left in the
        output buffer).

    Returns
    -------
    io.TextIOBase

        This is the output stream object where all data is sent.

    """

    def __init__(self, output=None, capture_fd=False):
        if output is None:
            output = io.StringIO()
        self.output = output
        self.old = None
        self.tee = None
        self.capture_fd = capture_fd
        self.context_stack = []

    def _enter_context(self, cm, prior_to=None):
        """Add the context manager to the context stack and return the result
        from calling the context manager's `__enter__()`

        """
        if prior_to is None:
            self.context_stack.append(cm)
        else:
            self.context_stack.insert(self.context_stack.index(prior_to), cm)
        return cm.__enter__()

    def _exit_context_stack(self, et, ev, tb):
        """Flush the context stack, calling __exit__() on all context managers

        One would usually use the contextlib.ExitStack to implement/manage
        the collection of context managers we are putting together.  The
        problem is that ExitStack will only call the __exit__ handlers
        up to the first one that returns an exception.  As we are
        expecting the possibility of one of the CMs here to raise an
        exception (usually from TeeStream when joining the reader
        threads), we will explicitly implement the stack management here
        so that we will guarantee that all __exit__ handlers will always
        be called.

        """
        FAIL = []
        while self.context_stack:
            try:
                self.context_stack.pop().__exit__(et, ev, tb)
            except:
                FAIL.append(str(sys.exc_info()[1]))
        return FAIL

    def __enter__(self):
        self.old = (sys.stdout, sys.stderr)
        old_fd = []
        for stream in self.old:
            stream.flush()
            try:
                old_fd.append(stream.fileno())
            except (AttributeError, OSError):
                old_fd.append(None)
        try:
            # We have an issue where we are (very aggressively)
            # commandeering the terminal.  This is what we intend, but the
            # side effect is that any errors generated by this module (e.g.,
            # because the user gave us an invalid output stream) get
            # completely suppressed.  So, we will make an exception to the
            # output that we are catching and let messages logged to THIS
            # logger to still be emitted to the original stderr.
            if self.capture_fd:
                # Because we are also commandeering the FD that underlies
                # sys.stderr, we cannot just write to that stream and
                # instead will open a new stream to the "original" FD
                # (Note that we need to duplicate that FD, as we will
                # overwrite it when we get to redirect_fd below).  If
                # sys.stderr doesn't have a file descriptor, we will
                # fall back on the process stderr (FD=2).
                log_stream = self._enter_context(
                    os.fdopen(os.dup(old_fd[1] or 2), mode="w", closefd=True)
                )
            else:
                log_stream = self.old[1]
            self._enter_context(LoggingIntercept(log_stream, logger=logger, level=None))

            if isinstance(self.output, str):
                self.output_stream = self._enter_context(open(self.output, 'w'))
            else:
                self.output_stream = self.output
            if isinstance(self.output, TeeStream):
                self.tee = self._enter_context(self.output)
            elif isinstance(self.output_stream, collections.abc.Sequence):
                self.tee = self._enter_context(TeeStream(*self.output_stream))
            else:
                self.tee = self._enter_context(TeeStream(self.output_stream))
            fd_redirect = {}
            if self.capture_fd:
                tee_fd = (self.tee.STDOUT.fileno(), self.tee.STDERR.fileno())
                for i in range(2):
                    # Redirect the standard process file descriptor (1 or 2)
                    fd_redirect[i + 1] = self._enter_context(
                        redirect_fd(i + 1, tee_fd[i], synchronize=False)
                    )
                    # Redirect the file descriptor currently associated with
                    # sys.stdout / sys.stderr
                    fd = old_fd[i]
                    if fd and fd not in fd_redirect:
                        fd_redirect[fd] = self._enter_context(
                            redirect_fd(fd, tee_fd[i], synchronize=False)
                        )
            # We need to make sure that we didn't just capture the FD
            # that underlies a stream that we are outputting to.  Note
            # that when capture_fd==False, normal streams will be left
            # alone, but the lastResort _StderrHandler() will still be
            # replaced (needed because that handler uses the *current*
            # value of sys.stderr)
            ostreams = []
            for stream in self.tee.ostreams:
                if isinstance(stream, LogStream):
                    for handler_redirect in stream.redirect_streams(fd_redirect):
                        self._enter_context(handler_redirect, prior_to=self.tee)
                else:
                    try:
                        fd = stream.fileno()
                    except (AttributeError, OSError):
                        fd = None
                    if fd in fd_redirect:
                        # We just redirected this file descriptor so
                        # we can capture the output.  This makes a
                        # loop that we really want to break.  Undo
                        # the redirect by pointing our output stream
                        # back to the original file descriptor.
                        stream = self._enter_context(
                            os.fdopen(
                                os.dup(fd_redirect[fd].original_fd),
                                mode="w",
                                closefd=True,
                            ),
                            prior_to=self.tee,
                        )
                ostreams.append(stream)
            self.tee.ostreams = ostreams
        except:
            # Note: we will ignore any exceptions raised while exiting
            # the context managers and just reraise the original
            # exception.
            self._exit_context_stack(*sys.exc_info())
            raise
        sys.stdout = self.tee.STDOUT
        sys.stderr = self.tee.STDERR
        buf = self.tee.ostreams
        if len(buf) == 1:
            buf = buf[0]
        return buf

    def __exit__(self, et, ev, tb):
        # Check that we were nested correctly
        FAIL = []
        if self.tee.STDOUT is not sys.stdout:
            FAIL.append('Captured output does not match sys.stdout.')
        if self.tee.STDERR is not sys.stderr:
            FAIL.append('Captured output does not match sys.stderr.')
        # Exit all context managers.  This includes
        #  - Restore any file descriptors we commandeered
        #  - Close / join the TeeStream
        #  - Close any opened files
        FAIL.extend(self._exit_context_stack(et, ev, tb))
        sys.stdout, sys.stderr = self.old
        self.old = None
        self.tee = None
        self.output_stream = None
        if FAIL:
            raise RuntimeError("\n".join(FAIL))

    def __del__(self):
        if self.tee is not None:
            self.__exit__(None, None, None)

    def setup(self):
        if self.old is not None:
            raise RuntimeError('Duplicate call to capture_output.setup.')
        return self.__enter__()

    def reset(self):
        return self.__exit__(None, None, None)


class _StreamHandle(object):
    def __init__(self, mode, buffering, encoding, newline):
        self.buffering = buffering
        self.newlines = newline
        self.flush = False
        self.read_pipe, self.write_pipe = os.pipe()
        if not buffering and 'b' not in mode:
            # While we support "unbuffered" behavior in text mode,
            # python does not
            buffering = -1
        self.write_file = os.fdopen(
            self.write_pipe,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            newline=newline,
            closefd=False,
        )
        if not self.buffering and buffering:
            # We want this stream to be unbuffered, but Python doesn't
            # allow it for text streams.  Mock up an unbuffered stream
            # using AutoFlush
            self.write_file = _AutoFlush(self.write_file, self)
        else:
            self.write_file = _SignalFlush(self.write_file, self)
        self.decoder_buffer = b''
        try:
            self.encoding = encoding or self.write_file.encoding
        except AttributeError:
            self.encoding = None
        if self.encoding:
            self.output_buffer = ''
        else:
            self.output_buffer = b''

    def fileno(self):
        return self.read_pipe

    def close(self):
        # Close both the file and the underlying file descriptor.  Note
        # that this may get called more than once.
        if self.write_file is not None:
            self.write_file.close()
            self.write_file = None

        if self.write_pipe is not None:
            # If someone else has closed the file descriptor, then
            # python raises an OSError
            try:
                os.close(self.write_pipe)
            except OSError:
                pass
            self.write_pipe = None

    def finalize(self, ostreams):
        self.decodeIncomingBuffer()
        if ostreams:
            self.writeOutputBuffer(ostreams, True)
        os.close(self.read_pipe)

        if self.output_buffer:
            logger.error(
                "Stream handle closed with a partial line "
                "in the output buffer that was not emitted to the "
                "output stream(s):\n\t'%s'" % (self.output_buffer,)
            )
        if self.decoder_buffer:
            logger.error(
                "Stream handle closed with un-decoded characters "
                "in the decoder buffer that was not emitted to the "
                "output stream(s):\n\t%r" % (self.decoder_buffer,)
            )

    def decodeIncomingBuffer(self):
        if not self.encoding:
            self.output_buffer, self.decoder_buffer = self.decoder_buffer, b''
            return

        raw_len = len(self.decoder_buffer)
        chars = ''
        while raw_len:
            try:
                chars = self.decoder_buffer[:raw_len].decode(self.encoding)
                break
            except:
                pass
            # partial read of unicode character, try again with
            # a shorter bytes buffer
            raw_len -= 1
        if self.newlines is None:
            chars = chars.replace('\r\n', '\n').replace('\r', '\n')
        self.output_buffer += chars
        self.decoder_buffer = self.decoder_buffer[raw_len:]

    def writeOutputBuffer(self, ostreams, flush):
        if not self.encoding:
            ostring, self.output_buffer = self.output_buffer, b''
        elif self.buffering > 0 and not flush:
            EOL = self.output_buffer.rfind(self.newlines or '\n') + 1
            ostring = self.output_buffer[:EOL]
            self.output_buffer = self.output_buffer[EOL:]
        else:
            ostring, self.output_buffer = self.output_buffer, ''

        if not ostring:
            return

        for stream in ostreams:
            try:
                written = stream.write(ostring)
            except:
                my_repr = "<%s.%s @ %s>" % (
                    stream.__class__.__module__,
                    stream.__class__.__name__,
                    hex(id(stream)),
                )
                if my_repr in ostring:
                    # In the case of nested capture_outputs, all the
                    # handlers are left on the logger.  We want to make
                    # sure that we don't create an infinite loop by
                    # re-printing a message *this* object generated.
                    continue
                et, e, tb = sys.exc_info()
                msg = "Error writing to output stream %s:\n    %s: %s\n" % (
                    my_repr,
                    et.__name__,
                    e,
                )
                if getattr(stream, 'closed', False):
                    msg += "Output stream closed before all output was written to it."
                else:
                    msg += "Is this a writeable TextIOBase object?"
                logger.error(
                    f"{msg}\nThe following was left in the output buffer:\n"
                    f"    {ostring!r}"
                )
                continue
            if flush or (written and not self.buffering):
                stream.flush()
            # Note: some derived file-like objects fail to return the
            # number of characters written (and implicitly return None).
            # If we get None, we will just assume that everything was
            # fine (as opposed to tossing the incomplete write error).
            if written is not None and written != len(ostring):
                my_repr = "<%s.%s @ %s>" % (
                    stream.__class__.__module__,
                    stream.__class__.__name__,
                    hex(id(stream)),
                )
                if my_repr in ostring[written:]:
                    continue
                logger.error(
                    "Incomplete write to output stream %s.\nThe following was "
                    "left in the output buffer:\n    %r" % (my_repr, ostring[written:])
                )


class TeeStream(object):
    def __init__(self, *ostreams, encoding=None, buffering=-1):
        self.ostreams = ostreams
        self.encoding = encoding
        self.buffering = buffering
        self._stdout = None
        self._stderr = None
        self._handles = []
        self._active_handles = []
        self._threads = []

    @property
    def STDOUT(self):
        if self._stdout is None:
            b = self.buffering
            if b == -1:
                b = 1
            self._stdout = self.open(buffering=b)
        return self._stdout

    @property
    def STDERR(self):
        if self._stderr is None:
            b = self.buffering
            if b == -1:
                b = 0
            self._stderr = self.open(buffering=b)
        return self._stderr

    def open(self, mode='w', buffering=-1, encoding=None, newline=None):
        if encoding is None:
            encoding = self.encoding
        handle = _StreamHandle(mode, buffering, encoding, newline)
        # Note that is it VERY important to close file handles in the
        # same thread that opens it.  If you don't you can see deadlocks
        # and a peculiar error ("libgcc_s.so.1 must be installed for
        # pthread_cancel to work"; see https://bugs.python.org/issue18748)
        #
        # To accomplish this, we will keep two handle lists: one is the
        # set of "active" handles that the (merged reader) thread is
        # using, and the other the list of all handles so the original
        # thread can close them after the reader thread terminates.
        if handle.buffering:
            self._active_handles.append(handle)
        else:
            # Unbuffered handles should appear earlier in the list so
            # that they get processed first
            self._active_handles.insert(0, handle)
        self._handles.append(handle)
        self._start(handle)
        return handle.write_file

    def close(self, in_exception=False):
        # Close all open handles.  Note that as the threads may
        # immediately start removing handles from the list, it is
        # important that we iterate over a copy of the list.
        for h in list(self._handles):
            h.close()

        # Join all stream processing threads
        _poll = _poll_interval
        FAIL = False
        while True:
            for th in self._threads:
                th.join(_poll)
            self._threads[:] = [th for th in self._threads if th.is_alive()]
            if not self._threads:
                break
            _poll *= 2
            if _poll_timeout <= _poll < 2 * _poll_timeout:
                if in_exception:
                    # We are already processing an exception: no reason
                    # to trigger another, nor to deadlock for an extended time
                    break
                logger.warning(
                    "Significant delay observed waiting to join reader "
                    "threads, possible output stream deadlock"
                )
            elif _poll >= _poll_timeout_deadlock:
                logger.error("TeeStream: deadlock observed joining reader threads")
                # Defer raising the exception until after we have
                # cleaned things up
                FAIL = True
                break

        for h in list(self._handles):
            h.finalize(self.ostreams)

        self._threads.clear()
        self._handles.clear()
        self._active_handles.clear()
        self._stdout = None
        self._stderr = None
        if FAIL:
            raise RuntimeError("TeeStream: deadlock observed joining reader threads")

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        self.close(et is not None)

    def __del__(self):
        # Implement __del__ to guarantee that file descriptors are closed
        # ... but only if we are not called by the GC in the handler thread
        if threading.current_thread() not in self._threads:
            self.close()

    def _start(self, handle):
        if not _peek_available:
            th = threading.Thread(target=self._streamReader, args=(handle,))
            th.daemon = True
            th.start()
            self._threads.append(th)
        elif not self._threads:
            th = threading.Thread(target=self._mergedReader)
            th.daemon = True
            th.start()
            self._threads.append(th)
        else:
            # The merged reader is already running... nothing additional
            # needs to be done
            pass

    def _streamReader(self, handle):
        while True:
            new_data = os.read(handle.read_pipe, io.DEFAULT_BUFFER_SIZE)
            if handle.flush:
                flush = True
                handle.flush = False
            else:
                flush = False
            if new_data:
                handle.decoder_buffer += new_data
            elif not flush:
                break

            # At this point, we have new data sitting in the
            # handle.decoder_buffer
            handle.decodeIncomingBuffer()
            # Now, output whatever we have decoded to the output streams
            handle.writeOutputBuffer(self.ostreams, flush)
        #
        # print("STREAM READER: DONE")

    def _mergedReader(self):
        noop = []
        handles = self._active_handles
        _poll = _poll_interval
        _fast_poll_ct = _poll_rampup
        new_data = ''  # something not None
        while handles:
            flush = False
            if new_data is None:
                # For performance reasons, we use very aggressive
                # polling at the beginning (_poll_interval) and then
                # ramp up to a much more modest polling interval
                # (_poll_rampup_limit) as the process runs and the
                # frequency of new data appearing on the pipe slows
                if _fast_poll_ct:
                    _fast_poll_ct -= 1
                    if not _fast_poll_ct:
                        _poll *= 10
                        if _poll < _poll_rampup_limit:
                            # reset the counter (to potentially increase
                            # the polling interval again)
                            _fast_poll_ct = _poll_rampup
            else:
                new_data = None
            if _mswindows:
                for handle in list(handles):
                    try:
                        if handle.flush:
                            flush = True
                            handle.flush = False
                        pipe = get_osfhandle(handle.read_pipe)
                        numAvail = PeekNamedPipe(pipe, 0)[1]
                        if numAvail:
                            result, new_data = ReadFile(pipe, numAvail, None)
                            handle.decoder_buffer += new_data
                            break
                    except:
                        handles.remove(handle)
                        new_data = ''  # not None so the poll interval doesn't increase
                if new_data is None and not flush:
                    # PeekNamedPipe is non-blocking; to avoid swamping
                    # the core, sleep for a "short" amount of time
                    time.sleep(_poll)
                    continue
            else:
                # Because we could be *adding* handles to the TeeStream
                # while the _mergedReader is running, we want to
                # periodically time out and update the list of handles
                # that we are waiting for.  It is also critical that we
                # send select() a *copy* of the handles list, as we see
                # deadlocks when handles are added while select() is
                # waiting
                ready_handles = select(list(handles), noop, noop, _poll)[0]
                if ready_handles:
                    handle = ready_handles[0]
                    new_data = os.read(handle.read_pipe, io.DEFAULT_BUFFER_SIZE)
                    if new_data:
                        handle.decoder_buffer += new_data
                    else:
                        handles.remove(handle)
                        new_data = ''  # not None so the poll interval doesn't increase
                else:
                    for handle in handles:
                        if handle.flush:
                            new_data = ''
                            break
                    else:
                        new_data = None
                        continue

                if handle.flush:
                    flush = True
                    handle.flush = False

            # At this point, we have new data sitting in the
            # handle.decoder_buffer
            handle.decodeIncomingBuffer()

            # Now, output whatever we have decoded to the output streams
            handle.writeOutputBuffer(self.ostreams, flush)
        #
        # print("MERGED READER: DONE")
