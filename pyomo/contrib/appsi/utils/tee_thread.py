from threading import Thread, Event
from time import sleep
import sys


class TeeThread(Thread):
    def __init__(self, filename):
        self.filename = filename
        self.event = Event()
        super(TeeThread, self).__init__()

    def run(self):
        f = open(self.filename, 'r')
        while True:
            lines = f.readlines()
            if len(lines) != 0:
                sys.stdout.write(''.join(lines))
            if self.event.is_set():
                break
            sleep(0.01)
        f.close()
