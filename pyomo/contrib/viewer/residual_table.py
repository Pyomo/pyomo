#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  ___________________________________________________________________________
#
#  This module was originally developed as part of the IDAES PSE Framework
#
#  Institute for the Design of Advanced Energy Systems Process Systems
#  Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2019, by the
#  software owners: The Regents of the University of California, through
#  Lawrence Berkeley National Laboratory,  National Technology & Engineering
#  Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
#  University Research Corporation, et al. All rights reserved.
#
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
A simple GUI viewer/editor for Pyomo models.
"""
__author__ = "John Eslick"

import os
import logging

from pyomo.common.fileutils import this_file_dir
from pyomo.common.flags import building_documentation
from pyomo.contrib.viewer.report import value_no_exception, get_residual

import pyomo.contrib.viewer.qt as myqt
import pyomo.environ as pyo

_log = logging.getLogger(__name__)


# This lets the file be imported when the Qt UI is not available (or
# when building docs), but you won't be able to use it
class _ResidualTableUI(object):
    pass


class _ResidualTable(object):
    pass


# Note that the classes loaded here have signatures that are not
# parsable by Sphinx, so we won't attempt to import them if we are
# building the API documentation.
if not building_documentation():
    mypath = this_file_dir()
    try:
        _ResidualTableUI, _ResidualTable = myqt.uic.loadUiType(
            os.path.join(mypath, "residual_table.ui")
        )
    except:
        pass


class ResidualTable(_ResidualTable, _ResidualTableUI):
    def __init__(self, ui_data):
        super().__init__()
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


class ResidualDataModel(myqt.QAbstractTableModel):
    def __init__(self, parent, ui_data):
        super().__init__(parent)
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
        self._items = list(
            self.ui_data.model.component_data_objects(pyo.Constraint, active=ac)
        )

    def sort(self):
        def _inactive_to_back(c):
            if c.active:
                return float("inf")
            else:
                return float("-inf")

        self._items.sort(
            key=lambda o: (
                o is None,
                (
                    get_residual(self.ui_data, o)
                    if get_residual(self.ui_data, o) is not None
                    and not isinstance(get_residual(self.ui_data, o), str)
                    else _inactive_to_back(o)
                ),
            ),
            reverse=True,
        )

    def rowCount(self, parent=myqt.QtCore.QModelIndex()):
        return len(self._items)

    def columnCount(self, parent=myqt.QtCore.QModelIndex()):
        return len(self.column)

    def data(
        self, index=myqt.QtCore.QModelIndex(), role=myqt.Qt.ItemDataRole.DisplayRole
    ):
        row = index.row()
        col = self.column[index.column()]
        if role == myqt.Qt.ItemDataRole.DisplayRole:
            o = self._items[row]
            if col == "name":
                return str(o)
            elif col == "residual":
                return get_residual(self.ui_data, o)
            elif col == "active":
                return o.active
            elif col == "ub":
                return value_no_exception(o.upper)
            elif col == "lb":
                return value_no_exception(o.lower)
            elif col == "value":
                try:
                    return self.ui_data.value_cache[o]
                except KeyError:
                    return None
        else:
            return None

    def headerData(self, i, orientation, role=myqt.Qt.ItemDataRole.DisplayRole):
        """
        Return the column headings for the horizontal header and
        index numbers for the vertical header.
        """
        if (
            orientation == myqt.Qt.Orientation.Horizontal
            and role == myqt.Qt.ItemDataRole.DisplayRole
        ):
            return self.column[i]
        return None
