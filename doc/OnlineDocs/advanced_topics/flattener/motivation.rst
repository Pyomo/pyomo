Motivation
==========

The ``pyomo.dae.flatten`` module was originally developed to assist with
dynamic optimization. A very common operation in dynamic or multi-period
optimization is to initialize all time-indexed variables to their values
at a specific time point. However, for variables indexed by time and
arbitrary other indexing sets, this is difficult to do in a way that does
does not depend on the variable we are initializing. Things get worse
when we consider that a time index can exist on a parent block rather
than the component itself.

By "reshaping" time-indexed variables in a model into references indexed
only by time, the ``flatten_dae_components`` function allows us to perform
operations that depend on knowledge of time indices without knowing
anything about the variables that we are operating on.

This "flattened representation" of a model turns out to be useful for
dynamic optimization in a variety of other contexts. Examples include
constructing a tracking objective function and plotting results.
This representation is also useful in cases where we want to preserve
indexing along more than one set, as in PDE-constrained optimization.
The ``flatten_components_along_sets`` function allows partitioning
components while preserving multiple indexing sets.
In such a case, time and space-indexed data for a given variable is useful
for purposes such as initialization, visualization, and stability analysis.
