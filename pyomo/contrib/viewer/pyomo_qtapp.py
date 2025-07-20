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

import pyomo.contrib.viewer.qt as myqt
from pyomo.contrib.viewer.pyomo_viewer import qtconsole_app, qtconsole_available


class QtApp(qtconsole_app.JupyterQtConsoleApp if qtconsole_available else object):
    _kernel_cmd_show_ui = """try:
    ui.show()
except NameError:
    try:
        model
    except NameError:
        model=None
    ui = get_mainwindow(model=model, ask_close=False)
ui.setWindowTitle('Pyomo Model Viewer -- {}')"""

    _kernel_cmd_hide_ui = """try:
    ui.hide()
except NameError:
    pass"""

    _kernel_cmd_import_qt_magic = r"%gui qt"

    _kernel_cmd_import_ui = "from pyomo.contrib.viewer.ui import get_mainwindow"

    _kernel_cmd_import_pyomo_env = "import pyomo.environ as pyo"

    def active_widget_name(self):
        current_widget = self.window.tab_widget.currentWidget()
        current_widget_index = self.window.tab_widget.indexOf(current_widget)
        return self.window.tab_widget.tabText(current_widget_index).replace("'", '"')

    def show_ui(self):
        kc = self.window.active_frontend.kernel_client
        kc.execute(
            self._kernel_cmd_show_ui.format(self.active_widget_name()), silent=True
        )

    def hide_ui(self):
        kc = self.window.active_frontend.kernel_client
        kc.execute(self._kernel_cmd_hide_ui, silent=True)

    def run_script(self, checked=False, filename=None):
        """Run a python script in the current kernel."""
        if filename is None:
            # Show a dialog box for user to select working directory
            filename = myqt.QtWidgets.QFileDialog.getOpenFileName(
                self.window,
                "Run Script",
                os.getcwd(),
                "py (*.py);;text (*.txt);;all (*)",
            )
            if filename[0]:  # returns a tuple of file and filter or ("","")
                filename = filename[0]
            else:
                filename = None
        # Run script if one was selected
        if filename is not None:
            kc = self.window.active_frontend.kernel_client
            kc.execute("%run {}".format(filename))

    def kernel_pyomo_init(self, kc):
        kc.execute(self._kernel_cmd_import_qt_magic, silent=True)
        kc.execute(self._kernel_cmd_import_ui, silent=True)
        kc.execute(self._kernel_cmd_import_pyomo_env, silent=False)

    def init_qt_elements(self):
        super().init_qt_elements()
        self.kernel_pyomo_init(self.widget.kernel_client)
        self.run_script_act = myqt.QAction("&Run Script...", self.window)
        self.show_ui_act = myqt.QAction("&Show Pyomo Model Viewer", self.window)
        self.hide_ui_act = myqt.QAction("&Hide Pyomo Model Viewer", self.window)
        self.window.file_menu.addSeparator()
        self.window.file_menu.addAction(self.run_script_act)
        self.window.view_menu.addSeparator()
        self.window.view_menu.addAction(self.show_ui_act)
        self.window.view_menu.addAction(self.hide_ui_act)
        self.window.view_menu.addSeparator()
        self.run_script_act.triggered.connect(self.run_script)
        self.show_ui_act.triggered.connect(self.show_ui)
        self.hide_ui_act.triggered.connect(self.hide_ui)

    def new_frontend_master(self):
        widget = super().new_frontend_master()
        self.kernel_pyomo_init(widget.kernel_client)
        return widget
