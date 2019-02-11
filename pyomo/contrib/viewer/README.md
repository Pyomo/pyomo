# Pyomo Model Tree Viewer/Editor

## Overview
This is an interactive tree viewer for Pyomo models.  You can inspect and change values, bound, fixed, and active attributes of Pyomo components.  It also calculates and displays constraint and named expression values.

## Using

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

## To Do

1. Self contained option
    - Start IPython kernel
    - Launch UI
    - Attach kernel to a Jupyter qt-console widget in UI
2. Searching
3. Sorting
4. Filtering
