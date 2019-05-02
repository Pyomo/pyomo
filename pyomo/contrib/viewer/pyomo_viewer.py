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

from pyomo.contrib.viewer.pyqt_4or5 import *
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.manager import QtKernelManager

def make_jupyter_widget_with_kernel():

    return jupyter_widget

class MainWindow(QMainWindow):
    """A window that contains a single Qt console."""
    def __init__(self):
        super().__init__()
        km = QtKernelManager(autorestart=False)
        km.start_kernel()
        kc = km.client()
        kc.start_channels()
        self.jupyter_widget = RichJupyterWidget()
        self.jupyter_widget.kernel_manager = km
        self.jupyter_widget.kernel_client = kc
        self.setCentralWidget(self.jupyter_widget)
        kc.execute("%gui qt", silent=True)
        time.sleep(3) # I need something better here, but I need to make sure
                      # the QApplication in the kernel has started before
                      # attempting to start the UI or it won't start
        kc.execute(
            "from pyomo.contrib.viewer.ui import get_mainwindow", silent=True)
        kc.execute("import pyomo.environ as pyo", silent=True)
        kc.execute("ui, model = get_mainwindow()", silent=True)

    def shutdown_kernel(self):
        print('Shutting down kernel...')
        self.jupyter_widget.kernel_client.stop_channels()
        self.jupyter_widget.kernel_manager.shutdown_kernel()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.aboutToQuit.connect(window.shutdown_kernel)
sys.exit(app.exec_())
