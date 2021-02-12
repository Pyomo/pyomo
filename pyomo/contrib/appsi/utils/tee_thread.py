from threading import Thread, Event
from time import sleep
import sys
import io


class TeeThread(Thread):
    def __init__(self, filename, stream_to_flush):
        self.filename = filename
        self.stream_to_flush: io.IOBase = stream_to_flush
        self.event = Event()
        super(TeeThread, self).__init__()

    def run(self):
        f = open(self.filename, 'r')
        while True:
            self.stream_to_flush.flush()
            lines = f.read()
            if len(lines) != 0:
                sys.stdout.write(lines)
            if self.event.is_set():
                self.stream_to_flush.flush()
                lines = f.read()
                if len(lines) != 0:
                    sys.stdout.write(lines)
                break
            sleep(0.01)
        f.close()
