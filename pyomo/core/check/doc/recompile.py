import pyinotify
import os
import sys

class Handler(pyinotify.ProcessEvent):
    def process_IN_MODIFY(self, event):
        if event.name.endswith('.tex'):
            print "Recompiling " + event.name + "..."
            os.popen('pdflatex ' + event.name).read()

wm = pyinotify.WatchManager()
notifier = pyinotify.Notifier(wm, Handler())
wm.add_watch(os.getcwd(), pyinotify.IN_MODIFY)
try:
    notifier.loop()
except pyinotify.NotifierError, err:
    print >>sys.stderr, err
