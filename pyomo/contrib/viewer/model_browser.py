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
from pyomo.core.base.param import ParamData
from pyomo.environ import (
    Block,
    BooleanVar,
    Var,
    Constraint,
    Param,
    Expression,
    value,
    units,
)

import pyomo.contrib.viewer.qt as myqt

_log = logging.getLogger(__name__)


# This lets the file be imported when the Qt UI is not available (or
# when building docs), but you won't be able to use it
class _ModelBrowserUI(object):
    pass


class _ModelBrowser(object):
    pass


# Note that the classes loaded here have signatures that are not
# parsable by Sphinx, so we won't attempt to import them if we are
# building the API documentation.
if not building_documentation():
    import sys

    mypath = this_file_dir()
    try:
        _ModelBrowserUI, _ModelBrowser = myqt.uic.loadUiType(
            os.path.join(mypath, "model_browser.ui")
        )
    except:
        pass


class LineEditCreator(myqt.QItemEditorCreatorBase):
    """
    Class to create editor widget for int and floats in a model view type object
    """

    def createWidget(self, parent):
        return myqt.QLineEdit(parent=parent)


class NumberDelegate(myqt.QItemDelegate):
    """
    Tree view item delegate. This is used here to change how items are edited.
    """

    def __init__(self, parent):
        super().__init__(parent=parent)
        factory = myqt.QItemEditorFactory()
        factory.registerEditor(myqt.QMetaType.Int, LineEditCreator())
        factory.registerEditor(myqt.QMetaType.Double, LineEditCreator())
        self.setItemEditorFactory(factory)

    def setModelData(self, editor, model, index):
        if isinstance(editor, myqt.QComboBox):
            value = editor.currentText()
        else:
            value = editor.text()
        a = model.column[index.column()]
        isinstance(index.internalPointer().get(a), bool)
        try:  # Recognize ints and floats.
            if value == "False" or value == "false":
                index.internalPointer().set(a, False)
            elif value == "True" or value == "true":
                index.internalPointer().set(a, True)
            elif "." in value or "e" in value or "E" in value:
                index.internalPointer().set(a, float(value))
            else:
                index.internalPointer().set(a, int(value))
        except:  # If not a valid number ignore
            pass


class ModelBrowser(_ModelBrowser, _ModelBrowserUI):
    def __init__(self, ui_data, standard="Var"):
        """
        Create a dock widget with a QTreeView of a Pyomo model.

        Args:
            ui_data: Contains model and ui information
            standard: A standard setup for different types of model components
                {"Var", "Constraint", "Param", "Expression"}
        """
        super().__init__()
        self.setupUi(self)
        # The default int and double spin boxes are not good for this
        # application.  So just use regular line edits.
        number_delegate = NumberDelegate(self)
        self.ui_data = ui_data
        self.ui_data.updated.connect(self.update_model)
        self.treeView.setItemDelegate(number_delegate)
        if standard == "Var":
            # This if block sets up standard views
            components = (Var, BooleanVar)
            columns = ["name", "value", "ub", "lb", "fixed", "stale", "units", "domain"]
            editable = ["value", "ub", "lb", "fixed"]
            self.setWindowTitle("Variables")
        elif standard == "Constraint":
            components = Constraint
            columns = ["name", "value", "ub", "lb", "residual", "active"]
            editable = ["active"]
            self.setWindowTitle("Constraints")
        elif standard == "Param":
            components = Param
            columns = ["name", "value", "mutable", "units"]
            editable = ["value"]
            self.setWindowTitle("Parameters")
        elif standard == "Expression":
            components = Expression
            columns = ["name", "value", "units"]
            editable = []
            self.setWindowTitle("Expressions")
        else:
            raise ValueError("{} is not a valid view type".format(standard))
        # Create a data model.  This is what translates the Pyomo model into
        # a tree view.
        datmodel = ComponentDataModel(
            self,
            ui_data=ui_data,
            columns=columns,
            components=components,
            editable=editable,
        )
        self.datmodel = datmodel
        self.treeView.setModel(datmodel)
        self.treeView.setColumnWidth(0, 400)
        # Selection behavior: select a whole row, can select multiple rows.
        self.treeView.setSelectionBehavior(
            myqt.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.treeView.setSelectionMode(
            myqt.QAbstractItemView.SelectionMode.ExtendedSelection
        )

    def refresh(self):
        added = self.datmodel._update_tree()
        self.datmodel.layoutChanged.emit()

    def update_model(self):
        self.datmodel.update_model()


class ComponentDataItem(object):
    """
    This is a container for a Pyomo component to be displayed in a model tree
    view.

    Args:
        parent: parent data item
        o: pyomo component object
        ui_data: a container for data, as of now mainly just pyomo model
    """

    def __init__(self, parent, o, ui_data):
        self.ui_data = ui_data
        self.data = o
        self.parent = parent
        self.children = []  # child items
        self.ids = {}
        self.get_callback = {
            "value": self._get_value_callback,
            "lb": self._get_lb_callback,
            "ub": self._get_ub_callback,
            "expr": self._get_expr_callback,
            "residual": self._get_residual_callback,
            "units": self._get_units_callback,
            "domain": self._get_domain_callback,
        }
        self.set_callback = {
            "value": self._set_value_callback,
            "lb": self._set_lb_callback,
            "ub": self._set_ub_callback,
            "active": self._set_active_callback,
            "fixed": self._set_fixed_callback,
        }

    @property
    def _cache_value(self):
        return self.ui_data.value_cache.get(self.data, None)

    @property
    def _cache_units(self):
        return self.ui_data.value_cache_units.get(self.data, None)

    def add_child(self, o):
        """Add a child data item"""
        item = ComponentDataItem(self, o, ui_data=self.ui_data)
        self.children.append(item)
        self.ids[id(o)] = item
        return item

    def get(self, a):
        """Get an attribute"""
        if a in self.get_callback:
            return self.get_callback[a]()
        else:
            try:
                return getattr(self.data, a)
            except:
                return None

    def set(self, a, val):
        """set an attribute"""
        if a in self.set_callback:
            return self.set_callback[a](val)
        else:
            try:
                return setattr(self.data, a, val)
            except:
                _log.exception("Can't set value of {}".format(a))
                return None

    def _get_expr_callback(self):
        if hasattr(self.data, "expr"):
            return str(self.data.expr)
        else:
            return None

    def _get_value_callback(self):
        if isinstance(self.data, ParamData):
            v = value_no_exception(self.data, div0="divide_by_0")
            # Check the param value for numpy float and int, sometimes numpy
            # values can sneak in especially if you set parameters from data
            # and for whatever reason numpy values don't display
            if isinstance(v, float):  # includes numpy float
                v = float(v)
            elif isinstance(v, int):  # includes numpy int
                v = int(v)
            return v
        elif isinstance(
            self.data, (Var._ComponentDataClass, BooleanVar._ComponentDataClass)
        ):
            v = value_no_exception(self.data)
            # Check the param value for numpy float and int, sometimes numpy
            # values can sneak in especially if you set parameters from data
            # and for whatever reason numpy values don't display
            if isinstance(v, float):  # includes numpy float
                v = float(v)
            elif isinstance(v, int):  # includes numpy int
                v = int(v)
            return v
        elif isinstance(self.data, (float, int)):
            return self.data
        else:
            return self._cache_value

    def _get_lb_callback(self):
        if isinstance(self.data, (Var._ComponentDataClass)):
            return self.data.lb
        elif hasattr(self.data, "lower"):
            return value_no_exception(self.data.lower, div0="Divide_by_0")
        else:
            return None

    def _get_ub_callback(self):
        if isinstance(self.data, (Var._ComponentDataClass)):
            return self.data.ub
        elif hasattr(self.data, "upper"):
            return value_no_exception(self.data.upper, div0="Divide_by_0")
        else:
            return None

    def _get_residual_callback(self):
        if isinstance(self.data, Constraint._ComponentDataClass):
            return get_residual(self.ui_data, self.data)
        else:
            return None

    def _get_units_callback(self):
        if isinstance(self.data, (Var, Var._ComponentDataClass)):
            return str(units.get_units(self.data))
        if isinstance(self.data, (Param, ParamData)):
            return str(units.get_units(self.data))
        return self._cache_units

    def _get_domain_callback(self):
        if isinstance(self.data, Var._ComponentDataClass):
            return str(self.data.domain)
        if isinstance(self.data, (BooleanVar, BooleanVar._ComponentDataClass)):
            return "BooleanVar"
        return None

    def _set_value_callback(self, val):
        if isinstance(
            self.data, (Var._ComponentDataClass, BooleanVar._ComponentDataClass)
        ):
            try:
                self.data.value = val
            except:
                return
        elif isinstance(self.data, (Var, BooleanVar)):
            try:
                for o in self.data.values():
                    o.value = val
            except:
                return
        elif isinstance(self.data, ParamData):
            if not self.data.parent_component().mutable:
                return
            try:
                self.data.value = val
            except:
                return
        elif isinstance(self.data, Param):
            if not self.data.parent_component().mutable:
                return
            try:
                for o in self.data.values():
                    o.value = val
            except:
                return

    def _set_lb_callback(self, val):
        if isinstance(self.data, (Var._ComponentDataClass)):
            try:
                self.data.setlb(val)
            except:
                return
        elif isinstance(self.data, Var):
            try:
                for o in self.data.values():
                    o.setlb(val)
            except:
                return

    def _set_ub_callback(self, val):
        if isinstance(self.data, (Var._ComponentDataClass)):
            try:
                self.data.setub(val)
            except:
                return
        elif isinstance(self.data, Var):
            try:
                for o in self.data.values():
                    o.setub(val)
            except:
                return

    def _set_active_callback(self, val):
        if not val or val in ["False", "false", "0", "f", "F"]:
            # depending on the version of Qt, you may see a combo box that
            # lets you select true/false or may be able to type the combo
            # box will return True or False, if you have to type could be
            # something else
            val = False
        else:
            val = True
        try:
            if val:
                self.data.activate()
            else:
                self.data.deactivate()
        except:
            return

    def _set_fixed_callback(self, val):
        if not val or val in ["False", "false", "0", "f", "F"]:
            # depending on the version of Qt, you may see a combo box that
            # lets you select true/false or may be able to type the combo
            # box will return True or False, if you have to type could be
            # something else
            val = False
        else:
            val = True
        try:
            if val:
                self.data.fix()
            else:
                self.data.unfix()
        except:
            return


class ComponentDataModel(myqt.QAbstractItemModel):
    """
    This is a data model to provide the tree structure and information
    to the tree viewer
    """

    def __init__(
        self,
        parent,
        ui_data,
        columns=["name", "value"],
        components=(Var, BooleanVar),
        editable=[],
    ):
        super().__init__(parent)
        self.column = columns
        self._col_editable = editable
        self.ui_data = ui_data
        self.components = components
        self.update_model()

    def update_model(self):
        self.rootItems = []
        self._create_tree(o=self.ui_data.model)

    def _update_tree(self, parent=None, o=None):
        """
        Check tree structure against the Pyomo model to add or delete
        components as needed. The arguments are to be used in the recursive
        function. Entering into this don't specify any args.
        """
        # Blocks are special they define the hierarchy of the model, so first
        # check for blocks. Other component can be handled together
        if o is None and len(self.rootItems) > 0:  # top level object (no parent)
            parent = self.rootItems[0]  # should be single root node for now
            o = parent.data  # start with root node
            for no in o.component_objects(descend_into=False):
                # This will traverse the whole Pyomo model tree
                self._update_tree(parent=parent, o=no)
            return
        elif o is None:  # if o is None, but no root nodes (when no model)
            return

        # past the root node go down here
        item = parent.ids.get(id(o), None)
        if item is not None:  # check if any children of item where deleted
            for i in item.children:
                try:
                    if i.data.parent_block() is None:
                        i.parent.children.remove(i)
                        del i.parent.ids[id(i.data)]
                        del i  # probably should descend down and delete stuff
                except AttributeError:
                    # Probably an element of an indexed immutable param
                    pass
        if isinstance(
            o, Block._ComponentDataClass
        ):  # single block or element of indexed block
            if item is None:
                item = self._add_item(parent=parent, o=o)
            for no in o.component_objects(descend_into=False):
                self._update_tree(parent=item, o=no)
        elif isinstance(o, Block):  # indexed block, so need to add elements
            if item is None:
                item = self._add_item(parent=parent, o=o)
            if (
                hasattr(o.index_set(), "is_constructed")
                and o.index_set().is_constructed()
            ):
                for key in sorted(o.keys()):
                    self._update_tree(parent=item, o=o[key])
        elif isinstance(o, self.components):  # anything else
            if item is None:
                item = self._add_item(parent=parent, o=o)
            if (
                hasattr(o.index_set(), "is_constructed")
                and o.index_set().is_constructed()
            ):
                for key in sorted(o.keys()):
                    if key == None:
                        break  # Single variable so skip
                    item2 = item.ids.get(id(o[key]), None)
                    if item2 is None:
                        item2 = self._add_item(parent=item, o=o[key])
                    item2._visited = True
        return

    def _create_tree(self, parent=None, o=None):
        """
        This create a model tree structure to display in a tree view.
        Args:
            parent: a ComponentDataItem underwhich to create a TreeItem
            o: A Pyomo component to add to the tree
        """
        # Blocks are special they define the hierarchy of the model, so first
        # check for blocks. Other component can be handled together
        if isinstance(
            o, Block._ComponentDataClass
        ):  # single block or element of indexed block
            item = self._add_item(parent=parent, o=o)
            for no in o.component_objects(descend_into=False):
                self._create_tree(parent=item, o=no)
        elif isinstance(o, Block):  # indexed block, so need to add elements
            item = self._add_item(parent=parent, o=o)
            if (
                hasattr(o.index_set(), "is_constructed")
                and o.index_set().is_constructed()
            ):
                for key in sorted(o.keys()):
                    self._create_tree(parent=item, o=o[key])
        elif isinstance(o, self.components):  # anything else
            item = self._add_item(parent=parent, o=o)
            if (
                hasattr(o.index_set(), "is_constructed")
                and o.index_set().is_constructed()
            ):
                for key in sorted(o.keys()):
                    if key == None:
                        break  # Single variable so skip
                    self._add_item(parent=item, o=o[key])

    def _add_item(self, parent, o):
        """
        Add a root item if parent is None, otherwise add a child
        """
        if parent is None:
            item = self._add_root_item(o)
        else:
            item = parent.add_child(o)
        return item

    def _add_root_item(self, o):
        """
        Add a root tree item
        """
        item = ComponentDataItem(None, o, ui_data=self.ui_data)
        self.rootItems.append(item)
        return item

    def parent(self, index):
        if not index.isValid():
            return myqt.QtCore.QModelIndex()
        item = index.internalPointer()
        if item.parent is None:
            return myqt.QtCore.QModelIndex()
        else:
            return self.createIndex(0, 0, item.parent)

    def index(self, row, column, parent=myqt.QtCore.QModelIndex()):
        if not parent.isValid():
            return self.createIndex(row, column, self.rootItems[row])
        parentItem = parent.internalPointer()
        return self.createIndex(row, column, parentItem.children[row])

    def columnCount(self, parent=myqt.QtCore.QModelIndex()):
        """
        Return the number of columns
        """
        return len(self.column)

    def rowCount(self, parent=myqt.QtCore.QModelIndex()):
        if not parent.isValid():
            return len(self.rootItems)
        return len(parent.internalPointer().children)

    def data(
        self, index=myqt.QtCore.QModelIndex(), role=myqt.Qt.ItemDataRole.DisplayRole
    ):
        if (
            role == myqt.Qt.ItemDataRole.DisplayRole
            or role == myqt.Qt.ItemDataRole.EditRole
        ):
            a = self.column[index.column()]
            return index.internalPointer().get(a)
        elif role == myqt.Qt.ItemDataRole.ToolTipRole:
            if self.column[index.column()] == "name":
                o = index.internalPointer()
                if isinstance(o.data, Constraint._ComponentDataClass):
                    return o.get("expr")
                else:
                    return o.get("doc")
        elif role == myqt.Qt.ItemDataRole.ForegroundRole:
            if isinstance(
                index.internalPointer().data, (Block, Block._ComponentDataClass)
            ):
                return myqt.QColor(myqt.QtCore.Qt.GlobalColor.black)
            else:
                return myqt.QColor(myqt.QtCore.Qt.GlobalColor.blue)
        else:
            return

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

    def flags(self, index=myqt.QtCore.QModelIndex()):
        if self.column[index.column()] in self._col_editable:
            return (
                myqt.Qt.ItemFlag.ItemIsEnabled
                | myqt.Qt.ItemFlag.ItemIsSelectable
                | myqt.Qt.ItemFlag.ItemIsEditable
            )
        else:
            return myqt.Qt.ItemFlag.ItemIsEnabled | myqt.Qt.ItemFlag.ItemIsSelectable
