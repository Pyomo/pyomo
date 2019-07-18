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
UI Tests
"""
import pyutilib.th as unittest
import time
import pytest

from pyomo.environ import *
from pyomo.contrib.viewer.qt import qt_available
from pyomo.contrib.viewer.qt import QtCore, QMessageBox
from pyomo.contrib.viewer.ui import get_mainwindow, ModelBrowser

def get_model():
    # Borrowed this test model from the trust region tests
    m = ConcreteModel()
    m.z = Var(range(3), domain=Reals, initialize=2.)
    m.x = Var(range(4), initialize=2.)
    m.x[1] = 1.0
    m.x[2] = 0.0
    m.x[3] = None

    m.b1 = Block()
    m.b1.e1 = Expression(expr=m.x[0] + m.x[1])
    m.b1.e2 = Expression(expr=m.x[0]/m.x[2])
    m.b1.e3 = Expression(expr=m.x[3]*m.x[1])
    m.b1.e4 = Expression(expr=log(m.x[2]))
    m.b1.e5 = Expression(expr=log(m.x[2] - 2))

    def blackbox(a,b):
        return sin(a-b)
    m.bb = ExternalFunction(blackbox)

    m.obj = Objective(
        expr=(m.z[0]-1.0)**2 + (m.z[0]-m.z[1])**2 + (m.z[2]-1.0)**2 \
            + (m.x[0]-1.0)**4 + (m.x[1]-1.0)**6 # + m.bb(m.x[0],m.x[1])
        )
    m.c1 = Constraint(expr=m.x[0] * m.z[0]**2 + m.bb(m.x[0],m.x[1]) == 2*sqrt(2.0))
    m.c2 = Constraint(expr=m.z[2]**4 * m.z[1]**2 + m.z[1] == 8+sqrt(2.0))
    m.c3 = Constraint(expr=m.x[1] == 3)
    m.c4 = Constraint(expr=0 == 3/m.x[2])
    m.c5 = Constraint(expr=0 == log(m.x[2]))
    m.c6 = Constraint(expr=0 == log(m.x[2]-4))
    m.c7 = Constraint(expr=0 == log(m.x[3]))
    m.p1 = Param(mutable=True, initialize=1)
    m.c8 = Constraint(expr = m.x[1] <= 1/m.p1)
    m.p1 = 0
    return m

def get_button(w, label):
    """Get a buttom in window w labeled label"""
    blist = w.buttons()
    for b in blist:
        if b.text().replace("&", "") == label:
            return b
    return None

def test_get_mainwindow(qtbot):
    m = get_model()
    mw, m = get_mainwindow(model=m, testing=True)
    assert(hasattr(mw, "menuBar"))
    assert(isinstance(mw.variables, ModelBrowser))
    assert(isinstance(mw.constraints, ModelBrowser))
    assert(isinstance(mw.expressions, ModelBrowser))
    assert(isinstance(mw.parameters, ModelBrowser))

def test_model_information(qtbot):
    m = get_model()
    mw, m = get_mainwindow(model=m, testing=True)
    qtbot.addWidget(mw)
    qtbot.keyClick(mw.menuBar(), "m", modifier=QtCore.Qt.AltModifier)
    qtbot.keyClick(mw.menuModel, "i")
    assert(isinstance(mw._dialog, QMessageBox))
    text = mw._dialog.text()
    mw._dialog.close()
    text = text.split("\n")
    assert(text[0].startswith("8")) # Active constraints
    assert(text[1].startswith("7")) # Active equalities
    assert(text[2].startswith("7")) # Free vars in active equalities
    assert(text[3].startswith("0")) # degrees of feedom
    # Main window has parts it is supposed to 
    assert(hasattr(mw, "menuBar"))
    assert(isinstance(mw.variables, ModelBrowser))
    assert(isinstance(mw.constraints, ModelBrowser))
    assert(isinstance(mw.expressions, ModelBrowser))
    assert(isinstance(mw.parameters, ModelBrowser))
