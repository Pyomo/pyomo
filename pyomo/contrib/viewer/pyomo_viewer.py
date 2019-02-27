from pyomo.contrib.viewer.ui import *

def main():
    if not can_containt_qtconsole or not qt_available:
        _log.error("Cannot import PyQt or qtconsole")
        sys.exit(1)
    # The code below is based on the example
    # https://github.com/ipython/ipykernel/blob/master/examples/embedding/inprocess_qtconsole.py
    app = guisupport.get_app_qt4() # qt4 is okay even though its Qt5!
    kernel_manager = QtInProcessKernelManager()
    kernel_manager.start_kernel()
    kernel = kernel_manager.kernel
    kernel.gui = 'qt'
    kernel_client = kernel_manager.client()
    kernel_client.start_channels()
    ui, model = get_mainwindow_nb(main=True)
    ui._qtconsole.kernel_manager = kernel_manager
    ui._qtconsole.kernel_client = kernel_client
    import pyomo.environ as pyo
    # push the ui, model, and pyomo.environ module as pyo to ipy env
    kernel.shell.push({"ui":ui, "model":model, "pyo":pyo})
    guisupport.start_event_loop_qt4(app)

if __name__ == "__main__":
    main()
