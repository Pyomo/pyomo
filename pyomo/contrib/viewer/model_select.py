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

import logging
import os

from pyomo.common.fileutils import this_file_dir
from pyomo.common.flags import building_documentation

import pyomo.contrib.viewer.qt as myqt
import pyomo.environ as pyo

_log = logging.getLogger(__name__)


# This lets the file be imported when the Qt UI is not available (or
# when building docs), but you won't be able to use it
class _ModelSelectUI(object):
    pass


class _ModelSelect(object):
    pass


# Note that the classes loaded here have signatures that are not
# parsable by Sphinx, so we won't attempt to import them if we are
# building the API documentation.
if not building_documentation():
    mypath = this_file_dir()
    try:
        _ModelSelectUI, _ModelSelect = myqt.uic.loadUiType(
            os.path.join(mypath, "model_select.ui")
        )
    except:
        pass


class ModelSelect(_ModelSelect, _ModelSelectUI):
    def __init__(self, parent, ui_data):
        super().__init__(parent)
        self.setupUi(self)
        self.ui_data = ui_data
        self.closeButton.clicked.connect(self.close)
        self.selectButton.clicked.connect(self.select_model)

    def select_model(self):
        items = self.tableWidget.selectedItems()
        if len(items) == 0:
            return
        self.ui_data.model_var_name_in_main = self.models[items[0].row()][1]
        self.ui_data.model = self.models[items[0].row()][0]
        self.close()

    def update_models(self):
        import __main__

        s = dir(__main__)
        keys = []
        for k in s:
            if isinstance(getattr(__main__, k), pyo.Block):
                keys.append(k)
        self.tableWidget.clearContents()
        self.tableWidget.setRowCount(len(keys))
        self.models = []
        for row, k in enumerate(sorted(keys)):
            model = getattr(__main__, k)
            item = myqt.QTableWidgetItem()
            item.setText(k)
            self.tableWidget.setItem(row, 0, item)
            item = myqt.QTableWidgetItem()
            try:
                item.setText(model.name)
            except:
                item.setText("None")
            self.tableWidget.setItem(row, 1, item)
            item = myqt.QTableWidgetItem()
            item.setText(str(type(model)))
            self.tableWidget.setItem(row, 2, item)
            self.models.append((model, k))
