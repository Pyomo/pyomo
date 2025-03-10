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
UI data objects for sharing data and settings between different parts of the UI.
"""
__author__ = "John Eslick"

import logging
from pyomo.common.collections import ComponentMap
from pyomo.contrib.viewer.qt import *
import pyomo.environ as pyo

_log = logging.getLogger(__name__)


class UIDataNoUi(object):
    """
    This is the UIData object minus the signals.  This is the base class for
    UIData.  The class is split this way for testing when PyQt is not available.
    """

    def __init__(self, model=None, model_var_name_in_main=None):
        """
        This class holds the basic UI setup, but doesn't depend on Qt. It
        shouldn't really be used except for testing when Qt is not available.

        Args:
            model: The Pyomo model to view
            model_var_name_in_main: if this is set, check that the model variable
                which points to a model object in __main__ has the same id when
                the UI is refreshed due to a command being executed in jupyter
                notebook or QtConsole, if not the same id, then update the model
                Since the model viewer is not necessarily pointed at a model in the
                __main__ namespace only set this if you want the model to auto
                update.  Since the model selector dialog lets you choose models
                from the __main__ namespace it sets this when you select a model.
                This is useful if you run a script repeatedly that replaces a model
                preventing you from looking at a previous version of the model.
        """
        super().__init__()
        self._model = None
        self.model_var_name_in_main = model_var_name_in_main
        self._begin_update = False
        self.value_cache = ComponentMap()
        self.value_cache_units = ComponentMap()
        self.begin_update()
        self.model = model
        self.end_update()

    def begin_update(self):
        """
        Lets the model setup be changed without emitting the updated signal
        until the end_update function is called.
        """
        self._begin_update = True

    def end_update(self, emit=True):
        """
        Sets the begin update flag to false.  Needs to be overloaded to also
        emit an update signal in the full UIData class
        """
        self._begin_update = False

    def emit_update(self):
        """
        Don't forget to overloaded this, not raising a NotImplementedError so
        tests can run without Qt
        """
        pass

    def emit_exec_refresh(self):
        """
        Don't forget to overloaded this, not raising a NotImplementedError so
        tests can run without Qt
        """
        pass

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
        self.value_cache = ComponentMap()
        self.value_cache_units = ComponentMap()
        self.emit_update()

    def calculate_constraints(self):
        for o in self.model.component_data_objects(pyo.Constraint, active=True):
            try:
                self.value_cache[o] = pyo.value(o.body, exception=False)
            except ZeroDivisionError:
                self.value_cache[o] = "Divide_by_0"
        self.emit_exec_refresh()

    def calculate_expressions(self):
        for o in self.model.component_data_objects(pyo.Expression, active=True):
            try:
                self.value_cache[o] = pyo.value(o, exception=False)
            except ZeroDivisionError:
                self.value_cache[o] = "Divide_by_0"
            try:
                self.value_cache_units[o] = str(pyo.units.get_units(o))
            except:
                # If units aren't obtainable for whatever reason, let it go.
                pass
        self.emit_exec_refresh()


if not available:

    class UIData(UIDataNoUi):
        pass

else:

    class UIData(UIDataNoUi, QtCore.QObject):
        updated = Signal()
        exec_refresh = Signal()

        def __init__(self, *args, **kwargs):
            """
            This class holds the basic UI setup

            Args:
                model: The Pyomo model to view
            """
            super().__init__(*args, **kwargs)

        def end_update(self, emit=True):
            """
            Start automatically emitting update signal again when properties
            are changed and emit update for changes made between begin_update
            and end_update
            """
            super().end_update(emit=emit)
            if emit:
                self.emit_update()

        def emit_update(self):
            if not self._begin_update:
                self.updated.emit()

        def emit_exec_refresh(self):
            self.exec_refresh.emit()
