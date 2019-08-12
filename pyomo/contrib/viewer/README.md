# Pyomo Model Tree Viewer/Editor

## Overview
This is an interactive tree viewer for Pyomo models.  You can inspect and change
values, bound, fixed, and active attributes of Pyomo components.  It also
calculates and displays constraint and named expression values.

## Installation

### Requirements

The model viewer has a few additional Python package requirements beyond the
standard Pyomo install.

The standard way to use the model viewer is from IPython or Jupyter. **PyQt5**
is required, and to use the stand-alone viewer, Jupyter **qtconsole** is
required.

### Install

The Pyomo install also installs the viewer modules.

## Using

### Opening from IPython

This works with IPython, Jupyter notebook, Jupyter qtconsole, and IDEs and
editors (e.g. Spyder) that provide an IPython/Jupyter console for running code.
The following example shows how to setup a model viewer.

```python
%gui qt  #Enables IPython's GUI event loop integration.
# Execute the above in its own cell and wait for it to finish before moving on.
from pyomo.contrib.viewer.ui import get_mainwindow
import pyomo.environ as pyo

model = pyo.ConcreteModel() # could import an existing model here
ui = get_mainwindow(model=model)

# Do model things, the viewer will stay in sync with the Pyomo model
```

**Note:** the ```%gui qt``` cell must be executed in its own cell and execution
must complete before running any other cells (you can't use "run all").

The model viewer adds a callback after each cell executes to update the viewer
in case components have been added or removed from the model. The model viewer
should always display the current state of the model except for calculated
items.  You must explicitly request that constraint and expression calculations
be updated, since for very large models the time required may be significant.

### Opening the Stand-Alone Version

Run ```pyomo model-viewer``` to get a stand-alone model viewer. The standalone
viewer is based on the example code at
https://github.com/jupyter/qtconsole/blob/master/examples/embed_qtconsole.py.
The viewer will start with an empty Pyomo ConcreteModel called ```model```. The
advantage of the stand-alone viewer is that it will automatically set up the
environment and start the UI, saving typing a few lines of code. It also has a
few menu items to help do common tasks. In the kernel, ``pyomo.environ`` is
imported as ```pyo```. An empty ConcreteModel is available as ```model``` and
linked to the viewer. To launch the model viewer select "Show/Start Model Viewer"
from the "View" menu in the qtconsole window. After launching the model viewer
it is available as ```ui```.  This provides a useful ability to script UI
actions. You can link the model viewer to other models.

### Setting the Model

To set the viewer to look at a new model run (the model does not need to be
  named model):

```python
ui.set_model(model)
```

There is also a model selector in the file menu, which looks for Pyomo blocks in
the ```__main__``` name space and allows you to select one to view.

You can have multiple models and switch the model viewer widgets between them
using ```ui.set_model(model)```, or the model selector.

# Controling the UI

You can interact with the UI through code. For example (assuming the UI object
is ```ui```) to expand or collapse the tree view for variables:

```python
ui.variables.treeView.expandAll()
ui.variables.treeView.collapseAll()
```

To close the UI:

```
ui.close()
```

You can even add widgets and customize the interface while it is running.

## To Do

1. More rigorous automated tests
2. Searching
3. Sorting
4. Filtering
5. Documentation
6. Save/Load
7. Additional useful subwindows (maybe data frame views and plotting for
  multiple runs)?
