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

import logging
import os

_log = logging.getLogger(__name__)

import pyomo.environ as pyo
from pyomo.contrib.viewer.qt import *

mypath = os.path.dirname(__file__)
try:
    _ModelSelectUI, _ModelSelect = \
        uic.loadUiType(os.path.join(mypath, "model_select.ui"))
except:
    # This lets the file still be imported, but you won't be able to use it
    class _ModelSelectUI(object):
        pass
    class _ModelSelect(object):
        pass

class ModelSelect(_ModelSelect, _ModelSelectUI):
    def __init__(self, ui_data, parent=None):
        super(ModelSelect, self).__init__(parent=parent)
        self.setupUi(self)
        self.ui_data = ui_data
        self.closeButton.clicked.connect(self.close)
        self.selectButton.clicked.connect(self.select_model)

    def select_model(self):
        items = self.tableWidget.selectedItems()
        if len(items) == 0:
            return
        self.ui_data.model = self.models[items[0].row()]
        self.close()

    def update_models(self):
        import __main__
        s = __main__.__dict__
        keys = []
        for k in s:
            if isinstance(s[k], pyo.Block):
                keys.append(k)
        self.tableWidget.clearContents()
        self.tableWidget.setRowCount(len(keys))
        self.models = []
        for row, k in enumerate(sorted(keys)):
            item = QTableWidgetItem()
            item.setText(k)
            self.tableWidget.setItem(row, 0, item)
            item = QTableWidgetItem()
            try:
                item.setText(s[k].name)
            except:
                item.setText("None")
            self.tableWidget.setItem(row, 1, item)
            item = QTableWidgetItem()
            item.setText(str(type(s[k])))
            self.tableWidget.setItem(row, 2, item)
            self.models.append(s[k])
