##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# This software is distributed under the 3-clause BSD License.
##############################################################################
"""
A simple GUI viewer/editor for Pyomo models.
"""
from __future__ import division, print_function, absolute_import

__author__ = "John Eslick"

import time
import os
import warnings
import logging
import threading
import datetime
import json
import sys
from IPython import get_ipython
from IPython.lib import guisupport
import pyomo.contrib.viewer.report as rpt
import pyomo.environ as pe

_log = logging.getLogger(__name__)

try: # this is for importing PyQt 4 or 5, looks a bit nasty, settle on pyqt5?
    from PyQt5 import QtCore
except:
    _log.exception("Cannot import PyQt5.QtCore")
    try:
        from PyQt4 import QtCore
    except:
        _log.exception("Cannot import PyQt4.QtCore")
        QtCore = None
    else:
        try:
            from PyQt4.QtGui import QFileDialog, QMessageBox
            from PyQt4 import uic
        except:
            _log.exception("Cannot import PyQt4")
            QtCore = None
else:
    try:
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        from PyQt5 import uic
    except:
        _log.exception("Cannot import PyQt5")
        QtCore = None

from pyomo.contrib.viewer.model_browser import ModelBrowser

try:
    from qtconsole.rich_ipython_widget import RichIPythonWidget
    from qtconsole.inprocess import QtInProcessKernelManager
    _can_containt_qtconsole = True
except:
    _can_containt_qtconsole = False

_mypath = os.path.dirname(__file__)
try:
    _MainWindowUI, _MainWindow = \
        uic.loadUiType(os.path.join(_mypath, "main.ui"))
except:
    _log.exception("Failed to load UI files.")
    # This lets the file still be imported, but you won't be able to use it
    # Allowing this to be imported will let some basic tests pass without PyQt
    class _MainWindowUI(object):
        pass
    class _MainWindow(object):
        pass

def get_mainwindow_nb(model=None, show=True, main=False):
    """
    Create a UI MainWindow.

    Args:
        model: A Pyomo model to work with
        show: show the window after it is created
        main: if true add a qtconsole to make this the main application
    """
    if model is None:
        model = pe.ConcreteModel()
    ui = MainWindow(model=model, main=main)
    get_ipython().events.register('post_execute', ui.refresh_on_execute)
    if show: ui.show()
    return ui, model

def setup_environment():
    """
    Setup the standard environment
    """
    lines = [
        "import pyomo.environ as pe",
        "from pyomo.environ import SolverFactory, TransformationFactory",
        "from pyomo.contrib.viewer.ui import get_mainwindow_nb"]
    lines = "\n".join(lines)
    print(lines)
    return(lines)


class UISetup(QtCore.QObject):
    updated = QtCore.pyqtSignal()
    def __init__(self, model=None):

        """
        This class holds the basic UI setup

        Args:
            model: The Pyomo model to view
        """
        super(UISetup, self).__init__()
        self._model = None
        self._begin_update = False
        self.begin_update()
        self.model = model
        self.end_update()

    def begin_update(self):
        """
        Lets the model setup be changed without emitting the updated signal
        until the end_update function is called.
        """
        self._begin_update = True

    def end_update(self, noemit=False):
        """
        Start automatically emitting update signal again when properties are
        changed and emit update for changes made between begin_update and
        end_update
        """
        self._begin_update = False
        if not noemit: self.emit_update()

    def emit_update(self):
        if not self._begin_update: self.updated.emit()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
        self.emit_update()

class MainWindow(_MainWindow, _MainWindowUI):
    def __init__(self, *args, **kwargs):
        model = self.model = kwargs.pop("model", None)
        main = self.model = kwargs.pop("main", None)
        flags = kwargs.pop("flags", 0)
        if kwargs.pop("ontop", False):
            kwargs[flags] = flags | QtCore.Qt.WindowStaysOnTopHint
        self.ui_setup = UISetup(model=model)
        self.ui_setup.updated.connect(self.update_model)

        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        if main and _can_containt_qtconsole:
            self._qtconsole = RichIPythonWidget(parent=self)
            self.setCentralWidget(self._qtconsole)
        elif main and not _can_containt_qtconsole:
            _log.error("Cannot create qtconsole widget")
            sys.exit(1)
        self._main = main

        # Create model browsers these are dock widgets
        uis = self.ui_setup
        self.vars = ModelBrowser(parent=self, standard="Var", ui_setup=uis)
        self.cons = ModelBrowser(parent=self, standard="Constraint", ui_setup=uis)
        self.params = ModelBrowser(parent=self, standard="Param", ui_setup=uis)
        self.exprs = ModelBrowser(parent=self, standard="Expression", ui_setup=uis)

        # Dock the wigetes allong the bottom and tabify them
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.vars)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.cons)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.params)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.exprs)
        self.tabifyDockWidget(self.vars, self.params)
        self.tabifyDockWidget(self.vars, self.exprs)
        self.tabifyDockWidget(self.vars, self.cons)
        self.vars.raise_()

        # Set menu actions (rembeber the menu items are defined in the ui file)
        # you can edit the menus in qt-designer
        self.action_Exit.triggered.connect(self.exit_action)
        self.action_toggle_variables.triggered.connect(self.vars.toggle)
        self.action_toggle_constraints.triggered.connect(self.cons.toggle)
        self.action_toggle_parameters.triggered.connect(self.params.toggle)
        self.action_toggle_expressions.triggered.connect(self.exprs.toggle)
        self.actionToggle_Always_on_Top.triggered.connect(self.toggle_always_on_top)
        self.actionInformation.triggered.connect(self.model_information)
        self.actionCalculateConstraints.triggered.connect(self.cons.calculate_all)
        self.actionCalculateExpressions.triggered.connect(self.exprs.calculate_all)

    def set_model(self, model):
        self.ui_setup.model = model

    def update_model(self):
        pass

    def model_information(self):
        """
        Put some useful model information into a message box

        Displays:
        * number of active equality constraints
        * number of free variables in equality constraints
        * degrees of freedom

        Other things that could be added
        * number of deactivated equalities
        * number of active inequality constraints
        * number of deactivated inequality constratins
        * number of free variables not appearing in active constraints
        * number of fixed variables not appearing in active constraints
        * number of free variables not appearing in any constraints
        * number of fixed variables not appearing in any constraints
        * number of fixed variables appearing in constraints
        """
        active_eq = rpt.count_equality_constraints(self.ui_setup.model)
        free_vars = rpt.count_free_variables(self.ui_setup.model)
        dof = free_vars - active_eq
        if dof == 1:
            doftext = "Degree"
        else:
            doftext = "Degrees"
        QMessageBox.information(
            self,
            "Model Information",
            "{}  -- Active equalities\n"
            "{}  -- Free variables in active equalities\n"
            "{}  -- {} of freedom\n"\
            .format(active_eq, free_vars, dof, doftext))

    def toggle_always_on_top(self):
        """
        This toggles the always on top hint.  Whether this has any effect after
        the main window is created probably depends on the system.
        """
        self.setWindowFlags(window.windowFlags() ^ QtCore.Qt.WindowStaysOnTopHint)

    def refresh_on_execute(self):
        """
        This is the call back function that happens when code is executed in the
        ipython kernel.  The main purpose of this right now it to refresh the
        UI display so that it matches the current state of the model.
        """
        self.vars.refresh()
        self.cons.refresh()
        self.exprs.refresh()
        self.params.refresh()

    def exit_action(self):
        """
        Selecting exit from the UI, triggers the close event on this mainwindow
        """
        self.close()

    def closeEvent(self, event):
        """
        Handle the close event by asking for confirmation
        """
        result = QMessageBox.question(self,
            "Exit?",
            "Are you sure you want to exit ?",
            QMessageBox.Yes| QMessageBox.No)
        if result == QMessageBox.Yes:
            if self._main:
                self._qtconsole.kernel_client.stop_channels()
                self._qtconsole.kernel_manager.shutdown_kernel()
                guisupport.get_app_qt4().exit()
            event.accept()
        else:
            event.ignore()

def main():
    if QtCore is None:
        _log.error("Cannot import PyQt")
        sys.exit(1)
    if not _can_containt_qtconsole:
        _log.error("Cannot import qtconsole")
        sys.exit(1)
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
    kernel.shell.push({"ui":ui, "model":model}) # push the ui and model vars
    guisupport.start_event_loop_qt4(app)

if __name__ == "__main__":
    main()
