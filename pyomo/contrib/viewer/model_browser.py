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
import warnings
import logging
import re

_log = logging.getLogger(__name__)

from pyomo.contrib.viewer.pyqt_4or5 import *

from pyomo.core.base.block import _BlockData
from pyomo.core.base.var import _VarData
from pyomo.core.base.constraint import _ConstraintData
from pyomo.core.base.expression import _ExpressionData
from pyomo.network.port import SimplePort
from pyomo.core.base.param import _ParamData
from pyomo.environ import Block, Var, Constraint, Param, Expression, value

mypath = os.path.dirname(__file__)
try:
    _ModelBrowserUI, _ModelBrowser = \
        uic.loadUiType(os.path.join(mypath, "model_browser.ui"))
except:
    # This lets the file still be imported, but you won't be able to use it
    class _ModelBrowserUI(object):
        pass
    class _ModelBrowser(object):
        pass


class ModelBrowser(_ModelBrowser, _ModelBrowserUI):
    def __init__(self, ui_setup, parent=None, standard="Var"):
        """
        Create a dock widdget with a QTreeView of a Pyomo model.

        Args:
            parent: parent widget
            ui_setup: Contains model, and may containt more in future
            standard: A standard setup for differnt types of model components
                {"Var", "Constraint", "Param", "Expression"}
        """
        super(ModelBrowser, self).__init__(parent=parent)
        self.setupUi(self)
        self.ui_setup = ui_setup
        self.ui_setup.updated.connect(self.update_model)
        if standard == "Var":
            # This if block sets up standard views
            components = Var
            columns =  ["name", "value", "ub", "lb", "fixed", "stale"]
            editable = ["value", "ub", "lb", "fixed"]
            self.setWindowTitle("Variables")
        elif standard == "Constraint":
            components = Constraint
            columns = ["name", "value", "ub", "lb", "residual", "active", "expr"]
            editable = ["active"]
            self.setWindowTitle("Constraints")
        elif standard == "Param":
            components = Param
            columns = ["name", "value", "_mutable"]
            editable = ["value"]
            self.setWindowTitle("Parameters")
        elif standard == "Expression":
            components = Expression
            columns = ["name", "value"]
            editable = []
            self.setWindowTitle("Expressions")
        else:
            raise Exception("Not a valid view type")
        # Create a data model.  This is what translates the Pyomo model into
        # a tree view.
        datmodel = ComponentDataModel(self, ui_setup=ui_setup,
                                      columns=columns, components=components,
                                      editable=editable)
        self.datmodel = datmodel
        self.treeView.setModel(datmodel)
        self.treeView.setColumnWidth(0,400)
        # Set selection behavior so you select a whole row, and can selection
        # multiple rows.  At some point want to update calculate to add options
        # to calculate selected rows
        self.treeView.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.treeView.setSelectionMode(QAbstractItemView.ExtendedSelection)

    def refresh(self):
        added = self.datmodel._update_tree()
        self.datmodel.layoutChanged.emit()

    def toggle(self):
        if self.isVisible():
            self.hide()
        else:
            self.show()

    def update_model(self):
        self.datmodel.update_model()

    def calculate_all(self):
        for i in self.datmodel.rootItems:
            i.calculate_children()

class ComponentDataItem(object):
    """
    This is a container for a Pyomo component to be displayed in a model tree
    view.

    Args:
        parent: parent data item
        o: pyomo component object
        ui_setup: a container for data, as of now mainly just pyomo model
    """
    def __init__(self, parent, o, ui_setup):
        self.ui_setup = ui_setup
        self.data = o
        self.parent = parent
        self.children = [] # child items
        self.ids = {}
        self.clear_cache() # cache is for calculated items
        self.get_callback = {
            "value": self._get_value_callback,
            "lb": self._get_lb_callback,
            "ub": self._get_ub_callback,
            "expr": self._get_expr_callback,
            "residual": self._get_residual}
        self.set_callback = {
            "value":self._set_value_callback,
            "lb":self._set_lb_callback,
            "ub":self._set_ub_callback,
            "active":self._set_active_callback,
            "fixed":self._set_fixed_callback}

    def data_items():
        """Iterate through children data items and this one"""
        for i in self.children:
            i.data_items()
        yield self

    def add_child(self, o):
        """Add a child data item"""
        item = ComponentDataItem(self, o, ui_setup=self.ui_setup)
        self.children.append(item)
        self.ids[id(o)] = item
        return item

    def clear_cache(self):
        """Clear chache for calcuated items"""
        self._cache_value = None
        self._cache_lb = None
        self._cache_ub = None

    def calculate_children(self):
        self.calculate()
        for i in self.children:
            i.calculate_children()

    def calculate(self):
        """Calculate items, applies to expressions and constraints"""
        if isinstance(self.data, _ExpressionData):
            try:
                self._cache_value = value(self.data, exception=False)
            except ZeroDivisionError:
                self._cache_value = "Divide_by_0"
        if isinstance(self.data, _ConstraintData) and self.data.active:
            try:
                self._cache_value = value(self.data.body, exception=False)
            except ZeroDivisionError:
                self._cache_value = "Divide_by_0"
            try:
                self._cache_lb = value(self.data.lower, exception=False)
            except ZeroDivisionError:
                self._cache_lb = "Divide_by_0"
            try:
                self._cache_ub = value(self.data.upper, exception=False)
            except ZeroDivisionError:
                self._cache_ub = "Divide_by_0"

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
                return None

    def _get_expr_callback(self):
        if hasattr(self.data, "expr"):
            return str(self.data.expr)
        else:
            return None

    def _get_value_callback(self):
        if isinstance(self.data, (_VarData, _ParamData, float, int)):
            return value(self.data, exception=False)
        else:
            return self._cache_value

    def _get_lb_callback(self):
        if isinstance(self.data, _VarData):
            return self.data.lb
        else:
            return self._cache_lb

    def _get_ub_callback(self):
        if isinstance(self.data, _VarData):
            return self.data.ub
        else:
            return self._cache_ub

    def _get_residual(self):
        v = self._cache_value
        if v is None:
            return
        if self._cache_lb is not None and v < self._cache_lb:
            r1 = self._cache_lb - v
        else:
            r1 = 0
        if self._cache_ub is not None and v > self._cache_ub:
            r2 = v - self._cache_ub
        else:
            r2 = 0
        return max(r1, r2)

    def _set_value_callback(self, val):
        if isinstance(self.data, _VarData):
            try:
                self.data.value = float(val)
            except:
                return
        elif isinstance(self.data, _ParamData):
            if not self.data._mutable: return
            try:
                self.data.value = float(val)
            except:
                return

    def _set_lb_callback(self, val):
        if isinstance(self.data, _VarData):
            try:
                self.data.setlb(float(val))
            except:
                return

    def _set_ub_callback(self, val):
        if isinstance(self.data, _VarData):
            try:
                self.data.setub(float(val))
            except:
                return

    def _set_active_callback(self, val):
        if not val or val == "False" or val == "false" or val == "0" or \
            val == "f" or val == "F":
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
        if not val or val == "False" or val == "false" or val == "0" or \
            val == "f" or val == "F":
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


class ComponentDataModel(QAbstractItemModel):
    """
    This is a data model to provide the tree structure and information
    to the tree viewer
    """
    def __init__(self, parent, ui_setup, columns=["name", "value"],
                 components=(Var,), editable=[]):
        super(ComponentDataModel, self).__init__(parent)
        self.column = columns
        self._col_editable = editable
        self.ui_setup = ui_setup
        self.components = components
        self.update_model()

    def update_model(self):
        self.rootItems = []
        self._create_tree(o=self.ui_setup.model)

    def _update_tree(self, parent=None, o=None):
        """
        Check tree structure against the Pyomo model to add or delete
        components as needed. The arguments are to be used in the recursive
        function. Entering into this don't specify any args.
        """
        # Blocks are special they define the hiarchy of the model, so first
        # check for blocks. Other comonent can be handled togeter
        if o is None and len(self.rootItems) > 0: #top level object (no parent)
            parent = self.rootItems[0] # should be single root node for now
            o = parent.data # start with root node
            for no in o.component_objects(descend_into=False):
                # This will traverse the whole Pyomo model tree
                self._update_tree(parent=parent, o=no)
            return
        elif o is None: # if o is None, but no root nodes (when no model)
            return

        # past the root node go down here
        item = parent.ids.get(id(o), None)
        if item is not None: # check if any children of item where deleted
            for i in item.children:
                try:
                    if i.data.parent_block() is None:
                        i.parent.children.remove(i)
                        del(i.parent.ids[id(i.data)])
                        del(i) # probably should descend down and delete stuff
                except AttributeError:
                    # Probably an element of an indexed immutable param
                    pass
        if isinstance(o, _BlockData): #single block or element of indexed block
            if item is None:
                item = self._add_item(parent=parent, o=o)
            for no in o.component_objects(descend_into=False):
                self._update_tree(parent=item, o=no)
        elif isinstance(o, Block): #indexed block, so need to add elements
            if item is None:
                item = self._add_item(parent=parent, o=o)
            for key in sorted(o.keys()):
                self._update_tree(parent=item, o=o[key])
        elif isinstance(o, self.components): #anything else
            if item is None:
                item = self._add_item(parent=parent, o=o)
            for key in sorted(o.keys()):
                if key == None: break # Single variable so skip
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
        # Blocks are special they define the hiarchy of the model, so first
        # check for blocks. Other comonent can be handled togeter
        if isinstance(o, _BlockData): #single block or element of indexed block
            item = self._add_item(parent=parent, o=o)
            for no in o.component_objects(descend_into=False):
                self._create_tree(parent=item, o=no)
        elif isinstance(o, Block): #indexed block, so need to add elements
            item = self._add_item(parent=parent, o=o)
            for key in sorted(o.keys()):
                self._create_tree(parent=item, o=o[key])
        elif isinstance(o, self.components): #anything else
            item = self._add_item(parent=parent, o=o)
            for key in sorted(o.keys()):
                if key == None: break #Single variable so skip
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
        item = ComponentDataItem(None, o, ui_setup=self.ui_setup)
        self.rootItems.append(item)
        return item

    def parent(self, index):
        if not index.isValid():
            return QtCore.QModelIndex()
        item = index.internalPointer()
        if item.parent is None:
            return QtCore.QModelIndex()
        else:
            return self.createIndex(0, 0, item.parent)

    def index(self, row, column, parent=QtCore.QModelIndex()):
        if not parent.isValid():
            return self.createIndex(row, column, self.rootItems[row])
        parentItem = parent.internalPointer()
        return self.createIndex(row, column, parentItem.children[row])

    def columnCount(self, parent=QtCore.QModelIndex()):
        """
        Return the number of columns
        """
        return len(self.column)

    def rowCount(self, parent=QtCore.QModelIndex()):
        if not parent.isValid():
            return len(self.rootItems)
        return len(parent.internalPointer().children)

    def data(self, index=QtCore.QModelIndex(), role=QtCore.Qt.DisplayRole):
        if role==QtCore.Qt.DisplayRole or role==QtCore.Qt.EditRole:
            a = self.column[index.column()]
            return index.internalPointer().get(a)
        elif role==QtCore.Qt.ToolTipRole:
            if self.column[index.column()] == "name":
                o = index.internalPointer()
                if isinstance(o.data, _ConstraintData):
                    return o.get("expr")
                else:
                    return o.get("doc")
        else:
            return

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        if role==QtCore.Qt.EditRole:
            a = self.column[index.column()]
            if a in self._col_editable:
                index.internalPointer().set(a, value)
        return 1

    def headerData(self, i, orientation, role=QtCore.Qt.DisplayRole):
        """
        Return the column headings for the horizontal header and
        index numbers for the vertical header.
        """
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self.column[i]
        return None

    def flags(self, index=QtCore.QModelIndex()):
        if self.column[index.column()] in self._col_editable:
            return(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable |
                   QtCore.Qt.ItemIsEditable)
        else:
            return(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
