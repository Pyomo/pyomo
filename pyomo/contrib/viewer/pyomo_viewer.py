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
from pyomo.contrib.viewer.pyqt_4or5 import *

qtconsole_available = False
if qt_available:
    try:
        from qtconsole.rich_jupyter_widget import RichJupyterWidget
        from qtconsole.manager import QtKernelManager
        qtconsole_available = True
    except ImportError:
        pass

if qtconsole_available:
    class MainWindow(QMainWindow):
        """A window that contains a single Qt console."""
        def __init__(self):
            super().__init__()
            km = QtKernelManager(autorestart=False)
            km.start_kernel()
            self.execution_state = "starting"
            kc = km.client()
            kc.start_channels()
            kc.iopub_channel.message_received.connect(self.update_kernel_status)
            self.jupyter_widget = RichJupyterWidget()
            self.jupyter_widget.kernel_manager = km
            self.jupyter_widget.kernel_client = kc
            self.setCentralWidget(self.jupyter_widget)

        def update_kernel_status(self, msg):
            pass

        def shutdown_kernel(self):
            print('Shutting down kernel...')
            self.jupyter_widget.kernel_client.stop_channels()
            self.jupyter_widget.kernel_manager.shutdown_kernel()

def main(*args):
    if not qtconsole_available:
        print("qtconsole not available")
        return
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    kc = window.jupyter_widget.kernel_client
    kc.execute("%gui qt", silent=True)
    time.sleep(4.0) # can't find any other good to ensure qt finished startup
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
