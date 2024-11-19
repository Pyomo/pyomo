Pyomo leverages ``make`` to generate documentation.  The following two
sections describe how to build and test the online documentation
locally.

> **NOTE**: All commands assume you are running from the *root Pyomo source directory*.


Preview Changes Locally
-----------------------

1. Install documentation dependencies (e.g., Sphinx, etc):

   ```bash
   $ pip install -e .[docs]
   ```
   
   > **NOTE**: You may get a warning about the `dot` command if you do
   > not have `graphviz` installed.


2. Build the documentation.  Sphinx (and Pyomo) support multiple
   documentation *targets*.  These instructions describe building the
   `html` target, but the same process applies for other targets.

   ```bash
   $ make -C doc/OnlineDocs html
   ```

3. View ``doc/OnlineDocs/_build/html/index.html`` in your browser

Test Changes Locally
--------------------

   ```bash
   $ make -C doc/OnlineDocs doctest
   ```

Rebuilding the documentation
----------------------------

Sphinx caches significant amounts of work at the end of a documentation
build.  However, if you are in the process of editing the documentation,
it may not correctly invalidate the cache.  You can purge the entire
cache with

   ```bash
   $ make -C doc/OnlineDocs clean
   ```

Combining steps
---------------

These steps can, of course, be combined into a single command:

   ```bash
   $ make -c doc/OnlineDocs clean html doctest
   ```

Documentation history
---------------------

The Pyomo online documentation went through a significant overhaul in
2024.  If you need to go back and look at the old documentation, the
following git hashes might be relevant:

   - [23fb726ce](https://github.com/Pyomo/pyomo/commit/23fb726ce0e092412081bd70e8a0370af46f6d0f):
     main (close to just) before the reorg was merged

   - [c157587fc](https://github.com/Pyomo/pyomo/commit/c157587fc9a03300b53879b99c1f350a26a9519f):
     reorg branch just before the `Archive` subdirectory was removed
