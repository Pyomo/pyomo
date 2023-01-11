#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
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
try PyQt5 if that doesn't work. If no compatable Qt Python interface is found,
use some dummy classes to allow some testing.
"""
__author__ = "John Eslick"

import logging
import enum

_log = logging.getLogger(__name__)

qt_available = False

class DummyQt(object):
    class ItemDataRole(enum.Enum):
        EditRole = 1
        DisplayRole = 2
        ToolTipRole = 3
        ForegroundRole = 4

class DummyQtCore(object):
    """
    A dummy QtCore class to allow some testing without PyQt
    """

    class QModelIndex(object):
        pass

    Qt = DummyQt


class DummyQAbstractItemModel(object):
    """
    A dummy QAbstractItemModel class to allow some testing without PyQt
    """

    def __init__(*args, **kwargs):
        pass


class DummyQAbstractTableModel(object):
    """
    A dummy QAbstractTableModel class to allow some testing without PyQt
    """

    def __init__(*args, **kwargs):
        pass


try:
    from PySide6 import QtCore

    qt_available = "PySide6"
except:
    _log.error("PySide6 could not be impoerted, trying PyQt5")
    try:
        from PyQt5 import QtCore

        qt_available = "PyQt5"
    except:
        _log.error("PyQt5 could not be impoerted")
        pass

if qt_available == "PyQt5":
    from PyQt5.QtWidgets import (
        QAbstractItemView,
        QFileDialog,
        QMainWindow,
        QMessageBox,
        QMdiArea,
        QApplication,
        QTableWidgetItem,
        QAction,
        QStatusBar,
        QLineEdit,
        QItemEditorFactory,
        QItemEditorCreatorBase,
        QStyledItemDelegate,
        QItemDelegate,
        QComboBox,
    )
    from PyQt5.QtGui import QColor
    from PyQt5.QtCore import QAbstractItemModel, QAbstractTableModel
    import PyQt5.QtCore as QtCore
    from PyQt5.QtCore import QMetaType, Qt, pyqtSignal as Signal

    from PyQt5 import uic

elif qt_available == "PySide6":
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QFileDialog,
        QMainWindow,
        QMessageBox,
        QMdiArea,
        QApplication,
        QTableWidgetItem,
        QStatusBar,
        QLineEdit,
        QItemEditorFactory,
        QItemEditorCreatorBase,
        QStyledItemDelegate,
        QItemDelegate,
        QComboBox,
    )

    from PySide6.QtGui import QColor, QAction
    from PySide6.QtCore import QAbstractItemModel, QAbstractTableModel
    import PySide6.QtCore as QtCore
    from PySide6.QtCore import Qt, Signal, QMetaType

    from PySide6 import QtUiTools as uic


if not qt_available:
    # Dummy classes allow some testing without PyQt
    QAbstractItemModel = DummyQAbstractItemModel
    QAbstractTableModel = DummyQAbstractTableModel
    QtCore = DummyQtCore
    Qt = DummyQt
