##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2019, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# This software is distributed under the 3-clause BSD License.
##############################################################################

# based on the example code at:
#  https://github.com/jupyter/qtconsole/blob/master/examples/embed_qtconsole.py

from __future__ import print_function

import sys
import time

from pyomo.scripting.pyomo_parser import add_subparser
from pyomo.contrib.viewer.qt import *

qtconsole_available = False
if qt_available:
    try:
        from qtconsole.rich_jupyter_widget import RichJupyterWidget
        from qtconsole.manager import QtKernelManager
        qtconsole_available = True
    except ImportError:
        pass

if qtconsole_available:
    def _start_kernel():
        km = QtKernelManager(autorestart=False)
        km.start_kernel()
        kc = km.client()
        kc.start_channels()
        kc.execute("%gui qt", silent=True)
        time.sleep(1.5)
        return km, kc

    class MainWindow(QMainWindow):
        """A window that contains a single Qt console."""
        def __init__(self, kernel_manager, kernel_client):
            super(MainWindow, self).__init__()
            self.jupyter_widget = RichJupyterWidget()
            self.jupyter_widget.kernel_manager = kernel_manager
            self.jupyter_widget.kernel_client = kernel_client
            kernel_client.hb_channel.kernel_died.connect(self.close)
            self.setCentralWidget(self.jupyter_widget)

        def shutdown_kernel(self):
            print('Shutting down kernel...')
            self.jupyter_widget.kernel_client.stop_channels()
            self.jupyter_widget.kernel_manager.shutdown_kernel()

def main(*args):
    if not qtconsole_available:
        print("qtconsole not available")
        return
    km, kc = _start_kernel()
    app = QApplication(sys.argv)
    window = MainWindow(kernel_manager=km, kernel_client=kc)
    window.show()
    kc = window.jupyter_widget.kernel_client
    time.sleep(2.5) # can't find any other good way to ensure Qt finished
                    # startup. Just making sure the cell execution finishes
                    # does not seems to be enough, and trying to check
                    # QtAppliction() != None before moving on also does not
                    # seem to work. 4 seconds may be too long, but I'm being
                    # careful. I split the time between waiting for the console
                    # window to show and waiting to start the model viewer so it
                    # may not seem to take so long to the user.
                    # May be related to:
                    # https://github.com/ipython/ipython/issues/5629
    kc.execute("""
from pyomo.contrib.viewer.ui import get_mainwindow
import pyomo.environ as pyo
ui, model = get_mainwindow()""", silent=True)
    app.aboutToQuit.connect(window.shutdown_kernel)
    app.exec_()

# Add a subparser for the download-extensions command
add_subparser(
    'model-viewer',
    func=main,
    help='Run the Pyomo model viewer',
    add_help=False,
    description='This runs the Pyomo model viewer'
)

if __name__ == "__main__":
    main()
