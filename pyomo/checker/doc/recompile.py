#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

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
