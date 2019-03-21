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
"""
Run a Pyomo model viewer with embeded Jupyter QtConsole
"""

import sys
import logging
import traceback
_log = logging.getLogger(__name__)
from pyomo.contrib.viewer.ui import *
import pyomo.environ as pyo

ui = None

def log_unhandled_exception(etype, evalue, etrace):
    """
    This will log an exception and show a message box about what went wrong, but
    let the application keep running.  It could be frustrating if an unhandled
    exception caused the program to immediatly terminate.
    """
    if etype == KeyboardInterrupt:
        if ui is not None:
            ui.close()
            return
        else:
            sys.exit()
    try:
        _log.critical("unhandled exception", exc_info=(etype, evalue, etrace))
        msgBox = QMessageBox()
        msgBox.setWindowTitle("Error")
        msgBox.setText("Unhandled Exception: Please report this error to the "
                       "developers. Save and exit as soon as possible.")
        msgBox.setInformativeText(
            ''.join(traceback.format_exception(etype, evalue, etrace)))
        msgBox.exec_()
    except Exception:
        _log.exception("problem logging unhandled exception")

def main():
    global ui
    if not can_containt_qtconsole or not qt_available:
        _log.error("Cannot import qtconsole")
        sys.exit(1)
    # The code below is based on the example
    # https://github.com/ipython/ipykernel/blob/master/examples/embedding/inprocess_qtconsole.py
    app = guisupport.get_app_qt4() # qt4 is okay even though its Qt5!
    kernel_manager = QtInProcessKernelManager()
    kernel_manager.start_kernel()
    kernel = kernel_manager.kernel
    kernel.gui = 'qt'
    kernel_client = kernel_manager.client()
    kernel_client.start_channels()
    ui, model = get_mainwindow_nb(main=True)
    ui._qtconsole.kernel_manager = kernel_manager
    ui._qtconsole.kernel_client = kernel_client
    # push the ui, model, and pyomo.environ module as pyo to ipy env
    kernel.shell.push({"ui":ui, "model":model, "pyo":pyo})
    # catch any exception that falls through to prevent abrupt termination
    sys.excepthook = log_unhandled_exception
    guisupport.start_event_loop_qt4(app)

if __name__ == "__main__":
    main()
