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

import os
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
        # make sure there is no possible way the user can start the model
        # viewer before the Qt Application in the kernel finishes starting
        time.sleep(1.0)
        # Now just do the standard imports of things you want to be available
        # and whatever we may want to do to set up the environment just create
        # an empty model, so you can start the model viewer right away.  You
        # can add to the model if you want to use it, or create a new one.
        kc.execute("""
from pyomo.contrib.viewer.ui import get_mainwindow
import pyomo.environ as pyo
model = pyo.ConcreteModel("Default Model")""", silent=True)
        return km, kc

    class MainWindow(QMainWindow):
        """A window that contains a single Qt console."""
        def __init__(self, kernel_manager, kernel_client):
            super(MainWindow, self).__init__()
            self.jupyter_widget = RichJupyterWidget()
            self.jupyter_widget.kernel_manager = kernel_manager
            self.jupyter_widget.kernel_client = kernel_client
            kernel_client.hb_channel.kernel_died.connect(self.close)
            kernel_client.iopub_channel.message_received.connect(self.mrcv)
            menubar = self.menuBar()
            run_script_act = QAction("&Run Script...", self)
            wdir_set_act = QAction("Set &Working Directory...", self)
            exit_act = QAction("&Exit", self)
            show_ui_act = QAction("&Start/Show Model Viewer", self)
            hide_ui_act = QAction("&Hide Model Viewer", self)
            exit_act.triggered.connect(self.close)
            show_ui_act.triggered.connect(self.show_ui)
            hide_ui_act.triggered.connect(self.hide_ui)
            wdir_set_act.triggered.connect(self.wdir_select)
            run_script_act.triggered.connect(self.run_script)
            file_menu = menubar.addMenu('&File')
            view_menu = menubar.addMenu('&View')
            file_menu.addAction(wdir_set_act)
            file_menu.addAction(run_script_act)
            file_menu.addAction(exit_act)
            view_menu.addAction(show_ui_act)
            view_menu.addAction(hide_ui_act)
            self.status_bar = QStatusBar()
            self.setStatusBar(self.status_bar)
            self.status_bar.show()
            self.setCentralWidget(self.jupyter_widget)
            self._ui_created = False

        def wdir_select(self, checked=False, wdir=None):
            """
            Change the current working directory.

            Args:
                wdir (str): if None show a dialog to select, otherwise try to
                    change to this path
                checked (bool): triggered signal sends this, but it is not used
            Returns:
                (str): new working directory path
            """
            if wdir is None:
                # Show a dialog box for user to select working directory
                wd = QFileDialog(self, 'Working Directory', os.getcwd())
                wd.setFileMode(QFileDialog.DirectoryOnly)
            if wd.exec_() == QFileDialog.Accepted:
                wdir = wd.selectedFiles()[0]
            else:
                wdir = None
            # Change directory if one was selected
            if wdir is not None:
                os.chdir(wdir)
                self.jupyter_widget.kernel_client.execute(
                    "%cd {}".format(wdir))
            return wdir

        def run_script(self, checked=False, filename=None):
            """
            Change the current working directory.

            Args:
                filename (str): if None show a dialog to select, otherwise try
                    to change run filename script
                checked (bool): triggered signal sends this, but it is not used
            Returns:
                (str): selected script file
            """
            if filename is None:
                # Show a dialog box for user to select working directory
                filename = QFileDialog.getOpenFileName(
                    self,
                    'Run Script',
                    os.getcwd(),
                    "py (*.py);;text (*.txt);;all (*)")
                if filename[0]: # returns a tuple of file and filter or ("","")
                    filename = filename[0]
                else:
                    filename = None
            # Run script if one was selected
            if filename is not None:
                self.jupyter_widget.kernel_client.execute(
                    "%run {}".format(filename))
            return filename

        def hide_ui(self):
            if self._ui_created:
                self.jupyter_widget.kernel_client.execute(
                    "ui.hide()", silent=True)

        def show_ui(self):
            kc = self.jupyter_widget.kernel_client
            if self._ui_created:
                kc.execute("ui.show()", silent=True)
            else:
                self._ui_created = True
                kc.execute("ui, model = get_mainwindow(model=model)",
                           silent=True)

        def shutdown_kernel(self):
            print('Shutting down kernel...')
            self.jupyter_widget.kernel_client.stop_channels()
            self.jupyter_widget.kernel_manager.shutdown_kernel()

        def mrcv(self, m):
            try:
                stat = m['content']['execution_state']
                if stat:
                    self.status_bar.showMessage("Kernel Status: {}".format(stat))
            except:
                pass

def main(*args):
    if not qtconsole_available:
        print("qtconsole not available")
        return
    km, kc = _start_kernel()
    app = QApplication(sys.argv)
    window = MainWindow(kernel_manager=km, kernel_client=kc)
    window.show()
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
