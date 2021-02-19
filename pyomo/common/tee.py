#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
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
import io
import logging
import os
import sys
import threading
import time
from six import StringIO

_mswindows = sys.platform.startswith('win')
_poll_interval = 0.1
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


class capture_output(object):
    """
    Drop-in substitute for PyUtilib's capture_output.
    Takes in a StringIO, file-like object, or filename and temporarily
    redirects output to a string buffer.
    """
    def __init__(self, output=None):
        if output is None:
            output = StringIO()
        self.output = output
        self.output_file = None
        self.old = None
        self.tee = None

    def __enter__(self):
        if isinstance(self.output, str):
            self.output_stream = open(self.output, 'w')
        else:
            self.output_stream = self.output
        self.old = (sys.stdout, sys.stderr)
        self.tee = TeeStream(self.output_stream)
        self.tee.__enter__()
        sys.stdout = self.tee.STDOUT
        sys.stderr = self.tee.STDERR
        return self.output_stream

    def __exit__(self, et, ev, tb):
        FAIL = self.tee.STDOUT is not sys.stdout
        self.tee.__exit__(et, ev, tb)
        if self.output_stream is not self.output:
            self.output_stream.close()
        sys.stdout, sys.stderr = self.old
        self.old = None
        self.tee = None
        self.output_stream = None
        if FAIL:
            raise RuntimeError('Captured output does not match sys.stdout.')

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
        self.read_pipe, self.write_pipe = os.pipe()
        if not buffering and 'b' not in mode:
            # While we support "unbuffered" behavior in text mode,
            # python does not
            buffering = -1
        self.write_file = os.fdopen(self.write_pipe, mode=mode,
                                    buffering=buffering, encoding=encoding,
                                    newline=newline, closefd=False)
        self.decoder_buffer = b''
        try:
            self.encoding = encoding or self.write_file.encoding
        except AttributeError:
            self.encoding = None
        if self.encoding:
            self.output_buffer = ''
        else:
            self.output_buffer = b''

    def __repr__(self):
        return "%s(%s)" % (self.buffering, id(self))

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
        self.close()
        self.decodeIncomingBuffer()
        if ostreams:
            # Turn off buffering for the final write
            self.buffering = 0
            self.writeOutputBuffer(ostreams)
        os.close(self.read_pipe)

        if self.output_buffer:
            logger.error(
                "Stream handle closed with a partial line "
                "in the output buffer that was not emitted to the "
                "output stream(s):\n\t'%s'" % (self.output_buffer,))
        if self.decoder_buffer:
            logger.error(
                "Stream handle closed with un-decoded characters "
                "in the decoder buffer that was not emitted to the "
                "output stream(s):\n\t%r" % (self.decoder_buffer,))

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

    def writeOutputBuffer(self, ostreams):
        if not self.encoding:
            ostring, self.output_buffer = self.output_buffer, b''
        elif self.buffering == 1:
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
                written = 0
            if written and not self.buffering:
                stream.flush()
            if written != len(ostring):
                logger.error(
                    "Output stream closed before all output was "
                    "written to it. The following was left in "
                    "the output buffer:\n\t%r" % (ostring[written:],))


class TeeStream(object):
    def __init__(self, *ostreams, encoding=None):
        self.ostreams = ostreams
        self.encoding = encoding
        self._stdout = None
        self._stderr = None
        self._handles = []
        self._threads = []

    @property
    def STDOUT(self):
        if self._stdout is None:
            self._stdout = self.open(buffering=1)
        return self._stdout

    @property
    def STDERR(self):
        if self._stderr is None:
            self._stderr = self.open(buffering=0)
        return self._stderr

    def open(self, mode='w', buffering=-1, encoding=None, newline=None):
        if encoding is None:
            encoding = self.encoding
        handle = _StreamHandle(mode, buffering, encoding, newline)
        if handle.buffering:
            self._handles.append(handle)
        else:
            # Unbuffered handles should appear earlier in the list so
            # that they get processed first
            self._handles.insert(0, handle)
        self._start(handle)
        return handle.write_file

    def close(self, in_exception=False):
        # Close all open handles.  Note that as the threads may
        # immediately start removing handles from the list, it is
        # important that we iterate over a copy of the list.
        for h in list(self._handles):
            h.close()

        # Join all stream processing threads
        join_iter = 1
        while True:
            for th in self._threads:
                th.join(_poll_interval*join_iter)
            self._threads[:] = [th for th in self._threads if th.is_alive()]
            if not self._threads:
                break
            join_iter += 1
            if join_iter == 10:
                if in_exception:
                    # We are already processing an exception: no reason
                    # to trigger another
                    break
                raise RuntimeError(
                    "TeeStream: deadlock observed joining reader threads")

        self._threads.clear()
        self._handles.clear()
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        self.close(et is not None)

    def __del__(self):
        # Implement __del__ to guarantee that file descriptors are closed
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
            if not new_data:
                handle.finalize(self.ostreams)
                break
            handle.decoder_buffer += new_data

            # At this point, we have new data sitting in the
            # handle.decoder_buffer
            handle.decodeIncomingBuffer()
            # Now, output whatever we have decoded to the output streams
            handle.writeOutputBuffer(self.ostreams)
        #
        #print("STREAM READER: DONE")

    def _mergedReader(self):
        noop = []
        handles = self._handles
        while handles:
            if _mswindows:
                new_data = None
                for handle in list(handles):
                    try:
                        pipe = get_osfhandle(handle.read_pipe)
                        numAvail = PeekNamedPipe(pipe, 0)[1]
                        if numAvail:
                            result, new_data = ReadFile(pipe, numAvail, None)
                            handle.decoder_buffer += new_data
                            break
                    except:
                        handle.finalize(self.ostreams)
                        handles.remove(handle)
                        new_data = None
                if new_data is None:
                    # PeekNamedPipe is non-blocking; to avoid swamping
                    # the core, sleep for a "short" amount of time
                    time.sleep(_poll_interval)
                    continue
            else:
                # Because we could be *adding* handles to the TeeStream
                # while the _mergedReader is running, we want to
                # periodically time out and update the list of handles
                # that we are waiting for.  It is also critical that we
                # send select() a *copy* of the handles list, as we see
                # deadlocks when handles are added while select() is
                # waiting
                ready_handles = select(
                    list(handles), noop, noop, _poll_interval)[0]
                if not ready_handles:
                    continue

                handle = ready_handles[0]
                new_data = os.read(handle.read_pipe, io.DEFAULT_BUFFER_SIZE)
                if not new_data:
                    handle.finalize(self.ostreams)
                    handles.remove(handle)
                    continue
                handle.decoder_buffer += new_data

            # At this point, we have new data sitting in the
            # handle.decoder_buffer
            handle.decodeIncomingBuffer()

            # Now, output whatever we have decoded to the output streams
            handle.writeOutputBuffer(self.ostreams)
        #
        #print("MERGED READER: DONE")
