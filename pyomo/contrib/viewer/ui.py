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
try:
    from IPython import get_ipython
except ImportError:
    def get_ipython():
        raise AttributeError("IPython not available")
import pyomo.contrib.viewer.report as rpt
import pyomo.environ as pe
from pyomo.kernel import ComponentMap

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

if not qt_available:
    _log.error("Qt is not available. Cannot create UI classes.")
    raise ImportError("Could not import PyQt4 or PyQt5")

def get_mainwindow(model=None, show=True, testing=False):
    """
    Create a UI MainWindow.

    Args:
        model: A Pyomo model to work with
        show: show the window after it is created
    Returns:
        (ui, model): ui is the MainWindow widget, and model is the linked Pyomo
            model.  If no model is provided a new ConcreteModel is created
    """
    if model is None:
        model = pe.ConcreteModel()
    ui = MainWindow(model=model, testing=testing)
    try:
        get_ipython().events.register('post_execute', ui.refresh_on_execute)
    except AttributeError:
        pass # not in ipy kernel, so is fine to not register callback
    if show: ui.show()
    return ui, model

def get_mainwindow_nb(model=None, show=True, testing=False):
    return get_mainwindow(model=model, show=show, testing=testing)


class UISetup(QtCore.QObject):
    updated = QtCore.pyqtSignal()
    exec_refresh = QtCore.pyqtSignal()
    def __init__(self, model=None):

        """
        This class holds the basic UI setup

        Args:
            model: The Pyomo model to view
        """
        super(UISetup, self).__init__()
        self._model = None
        self._begin_update = False
        self.value_cache = ComponentMap()
        self.begin_update()
        self.model = model
        self.end_update()

    def begin_update(self):
        """
        Lets the model setup be changed without emitting the updated signal
        until the end_update function is called.
        """
        self._begin_update = True

    def end_update(self, emit=True):
        """
        Start automatically emitting update signal again when properties are
        changed and emit update for changes made between begin_update and
        end_update
        """
        self._begin_update = False
        if emit: self.emit_update()

    def emit_update(self):
        if not self._begin_update: self.updated.emit()

    def emit_exec_refresh(self):
        self.exec_refresh.emit()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
        self.value_cache = ComponentMap()
        self.emit_update()

    def calculate_constraints(self):
        for o in self.model.component_data_objects(pe.Constraint, active=True):
            try:
                self.value_cache[o] = pe.value(o.body, exception=False)
            except ZeroDivisionError:
                self.value_cache[o] = "Divide_by_0"
        self.emit_exec_refresh()

    def calculate_expressions(self):
        for o in self.model.component_data_objects(pe.Expression, active=True):
            try:
                self.value_cache[o] = pe.value(o, exception=False)
            except ZeroDivisionError:
                self.value_cache[o] = "Divide_by_0"
        self.emit_exec_refresh()


class MainWindow(_MainWindow, _MainWindowUI):
    def __init__(self, *args, **kwargs):
        model = self.model = kwargs.pop("model", None)
        main = self.model = kwargs.pop("main", None)
        self.testing = kwargs.pop("testing", False)
        flags = kwargs.pop("flags", 0)
        if kwargs.pop("ontop", False):
            kwargs[flags] = flags | QtCore.Qt.WindowStaysOnTopHint
        self.ui_setup = UISetup(model=model)
        self.ui_setup.updated.connect(self.update_model)
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.setCentralWidget(self.mdiArea)

        self._refresh_list = []
        self.variables = None
        self.constraints = None
        self.expressions = None
        self.parameters = None

        self.variables_restart()
        self.expressions_restart()
        self.constraints_restart()
        self.parameters_restart()

        self.mdiArea.tileSubWindows()

        # Set menu actions (remember the menu items are defined in the ui file)
        # you can edit the menus in qt-designer
        self.action_Exit.triggered.connect(self.exit_action)
        self.ui_setup.exec_refresh.connect(self.refresh_on_execute)
        self.actionRestart_Variable_View.triggered.connect(
            self.variables_restart)
        self.actionRestart_Constraint_View.triggered.connect(
            self.constraints_restart)
        self.actionRestart_Parameter_View.triggered.connect(
            self.parameters_restart)
        self.actionRestart_Expression_View.triggered.connect(
            self.expressions_restart)
        self.actionInformation.triggered.connect(self.model_information)
        self.actionCalculateConstraints.triggered.connect(
            self.ui_setup.calculate_constraints)
        self.actionCalculateExpressions.triggered.connect(
            self.ui_setup.calculate_expressions)
        self.actionTile.triggered.connect(self.mdiArea.tileSubWindows)
        self.actionCascade.triggered.connect(self.mdiArea.cascadeSubWindows)
        self.actionTabs.triggered.connect(self.toggle_tabs)
        self._dialog = None #dialog displayed so can access it easier for tests
        self._dialog_test_button = None # button clicked on dialog in test mode

    def toggle_tabs(self):
        self.mdiArea.setViewMode(not self.mdiArea.viewMode())

    def _tree_restart(self, w, standard):
        """
        Start/Restart the variables window
        """
        try:
            self._refresh_list.remove(w)
        except ValueError: # not in list? that's okay
            pass
        try:
            try:
                self.mdiArea.removeSubWindow(w.parent())
            except RuntimeError: # user closed with "X" button
                pass
            del w
            w = None
        except AttributeError:
            pass
        w = ModelBrowser(standard=standard, ui_setup=self.ui_setup)
        self.mdiArea.addSubWindow(w)
        w.parent().show() # parent is now a MdiAreaSubWindow
        self._refresh_list.append(w)
        return w

    def variables_restart(self):
        self.variables = self._tree_restart(self.variables, "Var")

    def expressions_restart(self):
        self.expressions = self._tree_restart(self.expressions, "Expression")

    def parameters_restart(self):
        self.parameters = self._tree_restart(self.parameters, "Param")

    def constraints_restart(self):
        self.constraints = self._tree_restart(self.constraints, "Constraint")

    def set_model(self, model):
        self.ui_setup.model = model

    def update_model(self):
        """
        Play it safe by restarting all the tree view widgets when the model updates
        """
        self.variables_restart()
        self.expressions_restart()
        self.constraints_restart()
        self.parameters_restart()

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
        msg = QMessageBox()
        self._dialog = msg
        #msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Model Information")
        msg.setText(
"""{}  -- Active equalities
{}  -- Free variables in active equalities
{}  -- {} of freedom""".format(active_eq, free_vars, dof, doftext))
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setModal(False)
        msg.show()

    def refresh_on_execute(self):
        """
        This is the call back function that happens when code is executed in the
        ipython kernel.  The main purpose of this right now it to refresh the
        UI display so that it matches the current state of the model.
        """
        for w in self._refresh_list:
            try:
                w.refresh()
            except RuntimeError: # window closed by user pushing "X" button
                pass

    def exit_action(self):
        """
        Selecting exit from the UI, triggers the close event on this mainwindow
        """
        self.close()

    def closeEvent(self, event):
        """
        Handle the close event by asking for confirmation
        """
        msg = QMessageBox()
        self._dialog = msg
        msg.setIcon(QMessageBox.Question)
        msg.setText("Are you sure you want to exit?")
        msg.setWindowTitle("Exit?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        if self.testing: # don't even show dialog just pretend button clicked
            result = self._dialog_test_button
        else:
            result = msg.exec_()
        if result == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
