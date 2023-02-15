Preview Changes Locally
------------------------

1. Install Sphinx

   ```bash
   $ pip install sphinx sphinx_rtd_theme sphinx_copybutton
   ```
   
   **NOTE**: You may get a warning about the `dot` command if you do not have
   `graphviz` installed.

1. Build the documentation

   ```bash
   $ make html      # Option 1
   $ make latexpdf  # Option 2
   ```

   **NOTE**:  If the local python is not on your path, then you may need to 
   invoke `make` differently.  For example, using the PyUtilib `lbin` command:
   
   ```bash
   $ lbin make html
   ```

1. Preview your work

   ```bash
   $ cd _build/html
   $ open index.html
   ```
