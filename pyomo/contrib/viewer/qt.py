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
Try to import PySide6, which is the current official Qt 6 Python interface. Then,
try PyQt5 if that doesn't work. If no compatible Qt Python interface is found,
use some dummy classes to allow some testing.
"""
__author__ = "John Eslick"
import sys
import enum
import importlib

from pyomo.common.collections import Bunch
from pyomo.common.flags import building_documentation

# Supported Qt wrappers in preferred order
supported = ["PySide6", "PyQt6", "PyQt5"]
# Import errors encountered, delay logging for testing reasons
import_errors = []
# Set this to the Qt wrapper module is available
available = False

# You can only have one Qt interface loaded at a time.  If something
# like IPython has already loaded an interface, then we will use that
# one.
_loaded = list(filter(sys.modules.__contains__, supported))
if _loaded:
    supported = _loaded

for module_str in supported:
    try:
        qt_package = importlib.import_module(module_str)
        QtWidgets = importlib.import_module(f"{module_str}.QtWidgets")
        QtCore = importlib.import_module(f"{module_str}.QtCore")
        QtGui = importlib.import_module(f"{module_str}.QtGui")
        available = module_str
        break
    except Exception as e:
        qt_package = QtWidgets = QtCore = QtGui = None
        import_errors.append(f"{e}")

if not available:
    # If Qt is not available, we still want to be able to test as much
    # as we can, so add some dummy classes that allow for testing
    class Qt(object):
        class ItemDataRole(enum.Enum):
            EditRole = 1
            DisplayRole = 2
            ToolTipRole = 3
            ForegroundRole = 4

    class QtCore(object):
        """
        A dummy QtCore class to allow some testing without PyQt
        """

        class QModelIndex(object):
            pass

        Qt = Qt

    class QAbstractItemModel(object):
        """
        A dummy QAbstractItemModel class to allow some testing without PyQt
        """

        def __init__(*args, **kwargs):
            pass

    class QAbstractTableModel(object):
        """
        A dummy QAbstractTableModel class to allow some testing without PyQt
        """

        def __init__(*args, **kwargs):
            pass

    class QItemEditorCreatorBase(object):
        """
        A dummy QItemEditorCreatorBase class to allow some testing without PyQt
        """

        pass

    class QItemDelegate(object):
        """
        A dummy QItemDelegate class to allow some testing without PyQt
        """

        pass

else:
    QAbstractItemView = QtWidgets.QAbstractItemView
    QFileDialog = QtWidgets.QFileDialog
    QMainWindow = QtWidgets.QMainWindow
    QMdiArea = QtWidgets.QMdiArea
    QApplication = QtWidgets.QApplication
    QTableWidgetItem = QtWidgets.QTableWidgetItem
    QStatusBar = QtWidgets.QStatusBar
    QLineEdit = QtWidgets.QLineEdit
    QItemEditorFactory = QtWidgets.QItemEditorFactory
    QItemEditorCreatorBase = QtWidgets.QItemEditorCreatorBase
    QStyledItemDelegate = QtWidgets.QStyledItemDelegate
    QItemDelegate = QtWidgets.QItemDelegate
    QComboBox = QtWidgets.QComboBox
    QMessageBox = QtWidgets.QMessageBox
    QColor = QtGui.QColor
    QAbstractItemModel = QtCore.QAbstractItemModel
    QAbstractTableModel = QtCore.QAbstractTableModel
    Qt = QtCore.Qt
    QMetaType = Bunch()
    if available == "PySide6":
        from PySide6.QtGui import QAction
        from PySide6.QtCore import Signal
        from PySide6 import QtUiTools as uic

        QMetaType.Int = QtCore.QMetaType.Type.Int.value
        QMetaType.Double = QtCore.QMetaType.Type.Double.value
    elif available == "PyQt6":
        from PyQt6.QtGui import QAction
        from PyQt6.QtCore import pyqtSignal as Signal
        from PyQt6 import uic

        QMetaType.Int = QtCore.QMetaType.Type.Int.value
        QMetaType.Double = QtCore.QMetaType.Type.Double.value
    elif available == "PyQt5":
        from PyQt5.QtWidgets import QAction
        from PyQt5.QtCore import pyqtSignal as Signal
        from PyQt5 import uic

        QMetaType.Int = QtCore.QMetaType.Type.Int
        QMetaType.Double = QtCore.QMetaType.Type.Double
    else:
        raise RuntimeError(f"Unknown Qt engine: {available}")
    # Note that QAbstractTableModel and QAbstractItemModel have
    # signatures that are not parsable by Sphinx, so we will hide them
    # if we are building the API documentation.
    if building_documentation():

        class QAbstractItemModel(object):
            pass

        class QAbstractTableModel(object):
            pass
