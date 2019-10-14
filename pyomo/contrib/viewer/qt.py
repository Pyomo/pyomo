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
Import PyQt5 if available, then try PyQt4, then, if all else fails, use some
dummy classes to allow some testing. If anything fails to import, the excpetion
is logged.  That should make it clear exacly what's missing, but it could be a
little annoying it you are using PyQt4 or don't need jupyter qtconsole.  In the
future, will probably cut PyQt4 support, so it will be less of an issue.
"""
__author__ = "John Eslick"

import logging
_log = logging.getLogger(__name__)


class DummyQtCore(object):
    """
    A dummy QtCore class to allow some testing without PyQt
    """
    class QModelIndex(object):
        pass
    class Qt(object):
        class DisplayRole(object):
            pass
        class EditRole(object):
            pass

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

qt_available = False
qt_import_errors = []

try:
    from PyQt5 import QtCore
except:
    qt_import_errors.append("Cannot import PyQt5.QtCore")
    try:
        from PyQt4 import QtCore
    except:
        qt_import_errors.append("Cannot import PyQt4.QtCore")
    else:
        try:
            from PyQt4.QtGui import (QAbstractItemView, QFileDialog, QMainWindow,
                                     QMessageBox, QMdiArea, QApplication,
                                     QTableWidgetItem, QColor, QAction,
                                     QStatusBar, QLineEdit, QItemEditorFactory,
                                     QItemEditorCreatorBase, QStyledItemDelegate,
                                     QItemDelegate, QComboBox)
            from PyQt4.QtCore import (QAbstractItemModel, QAbstractTableModel,
                                      QVariant)
            import PyQt4.QtCore as QtCore
            from PyQt4 import uic
            qt_available = True
        except:
            qt_import_errors.append("Cannot import PyQt4")
else:
    try:
        from PyQt5.QtWidgets import (QAbstractItemView, QFileDialog, QMainWindow,
                                     QMessageBox, QMdiArea, QApplication,
                                     QTableWidgetItem, QAction, QStatusBar,
                                     QLineEdit, QItemEditorFactory,
                                     QItemEditorCreatorBase, QStyledItemDelegate,
                                     QItemDelegate, QComboBox)
        from PyQt5.QtGui import QColor
        from PyQt5.QtCore import (QAbstractItemModel, QAbstractTableModel,
                                  QVariant)
        import PyQt5.QtCore as QtCore
        from PyQt5 import uic
        qt_available = True
    except:
        qt_import_errors.append("Cannot import PyQt5")

if not qt_available:
    QAbstractItemModel = DummyQAbstractItemModel
    QAbstractTableModel = DummyQAbstractTableModel
    QtCore = DummyQtCore
