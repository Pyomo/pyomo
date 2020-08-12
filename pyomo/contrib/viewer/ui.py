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

import os
import logging
try:
    from IPython import get_ipython
except ImportError:
    def get_ipython():
        raise AttributeError("IPython not available")
import pyomo.contrib.viewer.report as rpt
import pyomo.environ as pyo

_log = logging.getLogger(__name__)

from pyomo.contrib.viewer.qt import *
from pyomo.contrib.viewer.model_browser import ModelBrowser
from pyomo.contrib.viewer.residual_table import ResidualTable
from pyomo.contrib.viewer.model_select import ModelSelect
from pyomo.contrib.viewer.ui_data import UIData

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
    for _err in qt_import_errors:
        _log.error(_err)
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
        model = pyo.ConcreteModel(name="Default")
    ui = MainWindow(model=model, testing=testing)
    try:
        get_ipython().events.register('post_execute', ui.refresh_on_execute)
    except AttributeError:
        pass # not in ipy kernel, so is fine to not register callback
    if show: ui.show()
    return ui, model

def get_mainwindow_nb(model=None, show=True, testing=False):
    return get_mainwindow(model=model, show=show, testing=testing)


class MainWindow(_MainWindow, _MainWindowUI):
    def __init__(self, *args, **kwargs):
        model = self.model = kwargs.pop("model", None)
        main = self.model = kwargs.pop("main", None)
        self.testing = kwargs.pop("testing", False)
        flags = kwargs.pop("flags", 0)
        self.ui_data = UIData(model=model)
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.setCentralWidget(self.mdiArea)

        self._refresh_list = []
        self.variables = None
        self.constraints = None
        self.expressions = None
        self.parameters = None
        self.residuals = None

        self.update_model()

        self.ui_data.updated.connect(self.update_model)
        # Set menu actions (remember the menu items are defined in the ui file)
        # you can edit the menus in qt-designer
        self.actionModel_Selector.triggered.connect(self.show_model_select)
        self.ui_data.exec_refresh.connect(self.refresh_on_execute)
        self.actionRestart_Variable_View.triggered.connect(
            self.variables_restart)
        self.actionRestart_Constraint_View.triggered.connect(
            self.constraints_restart)
        self.actionRestart_Parameter_View.triggered.connect(
            self.parameters_restart)
        self.actionRestart_Expression_View.triggered.connect(
            self.expressions_restart)
        self.actionRestart_Residual_Table.triggered.connect(
            self.residuals_restart)
        self.actionInformation.triggered.connect(self.model_information)
        self.actionCalculateConstraints.triggered.connect(
            self.ui_data.calculate_constraints)
        self.actionCalculateExpressions.triggered.connect(
            self.ui_data.calculate_expressions)
        self.actionTile.triggered.connect(self.mdiArea.tileSubWindows)
        self.actionCascade.triggered.connect(self.mdiArea.cascadeSubWindows)
        self.actionTabs.triggered.connect(self.toggle_tabs)
        self._dialog = None #dialog displayed so can access it easier for tests
        self._dialog_test_button = None # button clicked on dialog in test mode
        self.mdiArea.setViewMode(QMdiArea.TabbedView)

    def toggle_tabs(self):
        # Could use not here, but this is a little more future proof
        if self.mdiArea.viewMode() == QMdiArea.SubWindowView:
            self.mdiArea.setViewMode(QMdiArea.TabbedView)
        elif self.mdiArea.viewMode() == QMdiArea.TabbedView:
            self.mdiArea.setViewMode(QMdiArea.SubWindowView)
        else:
            # There are no other modes unless there is a change in Qt so pass
            pass

    def _tree_restart(self, w, cls=ModelBrowser, **kwargs):
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
        w = cls(**kwargs)
        self.mdiArea.addSubWindow(w)
        w.parent().show() # parent is now a MdiAreaSubWindow
        self._refresh_list.append(w)
        return w

    def variables_restart(self):
        self.variables = self._tree_restart(
            w=self.variables, standard="Var", ui_data=self.ui_data)

    def expressions_restart(self):
        self.expressions = self._tree_restart(
            w=self.expressions, standard="Expression", ui_data=self.ui_data)

    def parameters_restart(self):
        self.parameters = self._tree_restart(
            w=self.parameters, standard="Param", ui_data=self.ui_data)

    def constraints_restart(self):
        self.constraints = self._tree_restart(
            w=self.constraints, standard="Constraint", ui_data=self.ui_data)

    def residuals_restart(self):
        self.residuals = self._tree_restart(
            w=self.residuals, cls=ResidualTable, ui_data=self.ui_data)

    def set_model(self, model):
        self.ui_data.model = model

    def update_model(self):
        """
        Play it safe by restarting all the tree view widgets when the model updates
        """
        self.variables_restart()
        self.expressions_restart()
        self.constraints_restart()
        self.parameters_restart()
        self.mdiArea.setActiveSubWindow(self.variables.parent())
        self.toggle_tabs()
        self.toggle_tabs()

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
        active_eq = rpt.count_equality_constraints(self.ui_data.model)
        free_vars = rpt.count_free_variables(self.ui_data.model)
        cons = rpt.count_constraints(self.ui_data.model)
        dof = free_vars - active_eq
        if dof == 1:
            doftext = "Degree"
        else:
            doftext = "Degrees"
        msg = QMessageBox()
        msg.setStyleSheet("QLabel{min-width: 600px;}")
        self._dialog = msg
        #msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Model Information")
        msg.setText(
"""{} -- Active Constraints
{} -- Active Equalities
{} -- Free Variables
{} -- {} of Freedom""".format(cons, active_eq, free_vars, dof, doftext))
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

    def show_model_select(self):
        model_select = ModelSelect(parent=self, ui_data=self.ui_data)
        model_select.update_models()
        model_select.show()

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
        msg.setText("Are you sure you want to close this window?"
                    " You can reopen it with ui.show().")
        msg.setWindowTitle("Close?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        if self.testing: # don't even show dialog just pretend button clicked
            result = self._dialog_test_button
        else:
            result = msg.exec_()
        if result == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
