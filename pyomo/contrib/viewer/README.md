# Pyomo Model Tree Viewer/Editor

## Overview
This is an interactive tree viewer for Pyomo models.  You can inspect and change
values, bound, fixed, and active attributes of Pyomo components.  It also
calculates and displays constraint and named expression values. When used with
Jupyter, the graphical elements are run within the Jupyter kernel, so the UI can
be extended at runtime. 

## Installation

### Requirements

The model viewer has a few additional Python package requirements beyond the
standard Pyomo install.

The standard way to use the model viewer is from IPython or Jupyter. **Pyside6** or 
**PyQt5** is required, and to use the stand-alone viewer, Jupyter **qtconsole** is
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

If you are working in Jupyter notebook, Jupyter qtconsole, or other Jupyter-
based IDEs, and your model is in the __main__ namespace (this is the usual case),
you can specify the model by its variable name as below.  The advantage of this
is that if you replace the model with a new model having the same variable name,
the UI will automatically update without having to manually reset the model pointer.

```python
%gui qt  #Enables IPython's GUI event loop integration.
# Execute the above in its own cell and wait for it to finish before moving on.
from pyomo.contrib.viewer.ui import get_mainwindow
import pyomo.environ as pyo

model = pyo.ConcreteModel() # could import an existing model here
ui = get_mainwindow(model_var_name_in_main="model")

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
viewer is the standard Jupyter qtconsole app with a few minor modifications. The
file menu, contains a run script action. This will allow you to run a script
to build a Pyomo model (this is the same as using the %run magic). The view 
menu has two additional items to show and hide the Pyomo model viewer. When a 
new Jupyter kernel is started, it will automatically set up the Pyomo model 
viewer and ```import pyomo.environ as pyo```.

Once the Pyomo model viewer is opened, the model viewer main window object is
available in the kernel as ```ui```. You can interact with the UI through the
Qt API, allowing you to add or modify UI elements at run time. In the qtconsole
app you can run multiple kernels and have multiple model viewers open. 

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

# Controlling the UI

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

## Known Bugs

If you use the Qt interface in the qtconsole app, the automatic documentation
for Qt functions doesn't work properly.  To fix this, you need to ```import PySide6``` 
or ```import PyQt5``` as appropriate.

If you use the Qt API to modify the Pyomo modelviewer on the fly, it is 
fairly easy to crash the kernel by providing the wrong arguments to a 
function.  Use caution and if you have a UI modification you regularly
use, it is probably best to run a canned script.   
