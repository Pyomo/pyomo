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

from pyomo.contrib.viewer.pyqt_4or5 import *
from pyomo.contrib.viewer.model_browser import ModelBrowser

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
        if main and can_containt_qtconsole:
            self._qtconsole = RichIPythonWidget(parent=self)
            self.setCentralWidget(self._qtconsole)
        elif main and not can_containt_qtconsole:
            _log.error("Cannot create qtconsole widget")
            sys.exit(1)
        self._main = main

        # Create model browsers these are dock widgets
        uis = self.ui_setup
        self.variables = ModelBrowser(parent=self, standard="Var", ui_setup=uis)
        self.constraints = ModelBrowser(parent=self, standard="Constraint", ui_setup=uis)
        self.parameters = ModelBrowser(parent=self, standard="Param", ui_setup=uis)
        self.expressions = ModelBrowser(parent=self, standard="Expression", ui_setup=uis)

        # Dock the widgets along the bottom and tabify them
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.variables)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.constraints)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.parameters)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.expressions)
        self.tabifyDockWidget(self.variables, self.parameters)
        self.tabifyDockWidget(self.variables, self.expressions)
        self.tabifyDockWidget(self.variables, self.constraints)
        self.variables.raise_()

        # Set menu actions (remember the menu items are defined in the ui file)
        # you can edit the menus in qt-designer
        self.action_Exit.triggered.connect(self.exit_action)
        self.action_toggle_variables.triggered.connect(self.variables.toggle)
        self.action_toggle_constraints.triggered.connect(self.constraints.toggle)
        self.action_toggle_parameters.triggered.connect(self.parameters.toggle)
        self.action_toggle_expressions.triggered.connect(self.expressions.toggle)
        self.actionToggle_Always_on_Top.triggered.connect(self.toggle_always_on_top)
        self.actionInformation.triggered.connect(self.model_information)
        self.actionCalculateConstraints.triggered.connect(self.calculate_constraints)
        self.actionCalculateExpressions.triggered.connect(self.calculate_expressions)

    def calculate_constraints(self):
        self.constraints.calculate_all()
        self.refresh_on_execute()

    def calculate_expressions(self):
        self.expressions.calculate_all()
        self.refresh_on_execute()

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
        self.variables.refresh()
        self.constraints.refresh()
        self.expressions.refresh()
        self.parameters.refresh()

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
