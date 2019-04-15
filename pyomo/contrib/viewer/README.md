# Pyomo Model Tree Viewer/Editor

## Overview
This is an interactive tree viewer for Pyomo models.  You can inspect and change values, bound, fixed, and active attributes of Pyomo components.  It also calculates and displays constraint and named expression values.

## Installation

### Requirements

The model viewer has a few additional Python package requirements beyond the standard Pyomo install.

For using the model viewer from IPython or Jupyter notebook, **PyQt5** is required.  To use the stand-alone viewer, Jupyter **qtconsole** and **ipykernel** are also required.

**Note:**
At the time this the time this was written, a bug in the released version of ipykernel (5.1.0) requires a more recent development version to be installed from https://github.com/ipython/ipykernel.

### Install

The Pyomo install also installs the viewer modules. To run the stand-alone viewer, just copy the pyomo_viewer.py script to a location that is convenient to run.   

## Using

### Opening from IPython

This works with IPython and Jupyter Notebook.  This even works in IDEs and editors (e.g. Spyder) that provide and IPython console for running code.  The following example shows how to setup a model the the viewer.

```python
from pyomo.contrib.viewer.ui import get_mainwindow_nb
%gui qt  #Enables Ipython's GUI event loop integration.
from pyomo.environ import ConcreteModel, Var

model = ConcreteModel() # could import an existing model here
ui = get_mainwindow_nb(model=model)

# Do model things, the viewer will stay in sync with the Pyomo model
```

**Note:** in a Jupyter Notebook the ```%gui qt``` cell must be executed on its own and execution must complete before running any other cells (you can't use "run all").  This may be a bug in Jupyter Notebook.

The model viewer adds an IPython callback after each cell executes to update the viewer in case components have been added or removed from the model. The model viewer should always display the current state of the model except for calculated items.  You must explicitly request that calculations be updated, since for very large models the time required may be significant.

### Stand-Alone with Embedded Jupyter QtConsole

Run the "pyomo_viewer.py" script to get a stand alone model viewer with an embedded IPython console. The viewer will start with an empty Pyomo ConcreteModel called ```model```. To set the viewer to look at a new model run (the model does not need to be named model):

```python
ui.set_model(model)
```

You could have multiple models in and switch the model viewer widgets between them using ```ui.set_model(model)```.

**Note:** Some versions of ipykernel may have a bug where an error message "Execution Stopped" appears when trying to run commands after a command results in an exception. Versions of ipykernel > 5.1.0 should be okay.  Running exit in the IPython console may not work depending on the ipykernel version. Close the main UI window or select exit from the file menu to quit.

Inside the QtConsole ``pyomo.environ`` is imported as ```pyo```. An empty Concrete model is imported as ```model```. The main user interface window is imported as ```ui```.  This provides a useful ability to script ui actions.  You can even do things that are not yet possible through the graphical part of the user interface. For example to expand or collapse the tree view for variables:

```python
ui.variables.treeView.expandAll()
ui.variables.treeView.collapseAll()
```

To close the application:

```
ui.close()
```

Potentially you could even add widgets and customize the interface while it is running.

## To Do

1. More rigorous automated tests
2. Searching
3. Sorting
4. Filtering
5. Documentation
6. Save/Load
