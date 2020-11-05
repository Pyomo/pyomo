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

_log = logging.getLogger(__name__)

from pyomo.contrib.viewer.qt import *
from pyomo.contrib.viewer.report import value_no_exception, get_residual
import pyomo.environ as pyo

mypath = os.path.dirname(__file__)
try:
    _ResidualTableUI, _ResidualTable = \
        uic.loadUiType(os.path.join(mypath, "residual_table.ui"))
except:
    class _ResidualTableUI(object):
        pass
    class _ResidualTable(object):
        pass


class ResidualTable(_ResidualTable, _ResidualTableUI):
    def __init__(self, ui_data, parent=None):
        super(ResidualTable, self).__init__(parent=parent)
        self.setupUi(self)
        self.ui_data = ui_data
        datmodel = ResidualDataModel(parent=self, ui_data=ui_data)
        self.datmodel = datmodel
        self.tableView.setModel(datmodel)
        self.ui_data.updated.connect(self.refresh)
        self.sortButton.clicked.connect(self.sort)
        self.calculateButton.clicked.connect(self.calculate)

    def sort(self):
        self.datmodel.sort()

    def refresh(self):
        self.datmodel.update_model()
        self.datmodel.sort()
        self.datmodel.layoutChanged.emit()

    def calculate(self):
        self.ui_data.calculate_constraints()
        self.refresh()

class ResidualDataModel(QAbstractTableModel):
    def __init__(self, parent, ui_data):
        super(ResidualDataModel, self).__init__(parent)
        self.column = ["name", "residual", "value", "ub", "lb", "active"]
        self.ui_data = ui_data
        self.include_inactive = True
        self.update_model()
        self.sort()

    def update_model(self):
        if self.include_inactive:
            ac = None
        else:
            ac = True
        self._items = list(self.ui_data.model.component_data_objects(
             pyo.Constraint, active=ac))

    def sort(self):
        self._items.sort(key=
            lambda o: (o is None, get_residual(self.ui_data, o)
                                  if get_residual(self.ui_data, o) is not None
                                  else -float("inf")), reverse=True)

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._items)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.column)

    def data(self, index=QtCore.QModelIndex(), role=QtCore.Qt.DisplayRole):
        row = index.row()
        col = self.column[index.column()]
        if role == QtCore.Qt.DisplayRole:
            o = self._items[row]
            if col=="name":
                return str(o)
            elif col=="residual":
                return get_residual(self.ui_data, o)
            elif col=="active":
                return o.active
            elif col=="ub":
                return value_no_exception(o.upper)
            elif col=="lb":
                return value_no_exception(o.lower)
            elif col=="value":
                try:
                    return self.ui_data.value_cache[o]
                except KeyError:
                    return None
        else:
            return None

    def headerData(self, i, orientation, role=QtCore.Qt.DisplayRole):
        '''
            Return the column headings for the horizontal header and
            index numbers for the vertical header.
        '''
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self.column[i]
        return None
