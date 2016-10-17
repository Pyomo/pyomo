= A Simple Pyomo Script =

The `pyomo` command executes a simple workflow in which a model is created, a concrete instance is constructed from data, an optimizer is used to analyze the model, and optimization results are summarized.  This workflow can be easily executed within a Python script.

== Example ==

Suppose that the following Pyomo model is in the file `knapsack.py`:[[BR]]
[[Include(source:pyomo.data.samples/trunk/pyomo/data/samples/scripts/s1/knapsack.py)]]

Suppose that the following Pyomo data file is `knapsack.dat`:[[BR]]
[[Include(source:pyomo.data.samples/trunk/pyomo/data/samples/scripts/s1/knapsack.dat, text/x-text)]]

The following script executes a workflow that is similar to that executed by the `pyomo` command:[[BR]]
[[Include(source:pyomo.data.samples/trunk/pyomo/data/samples/scripts/s1/script.py)]]

This script generates the following output, which describes problem information, solver status, and the final solution:[[BR]]
[[Include(source:pyomo.data.samples/trunk/pyomo/data/samples/scripts/s1/script.out, text/x-text)]]


== Downloads ==

 * [source:pyomo.data.samples/trunk/pyomo/data/samples/scripts/s1/knapsack.py knapsack.py]
 * [source:pyomo.data.samples/trunk/pyomo/data/samples/scripts/s1/knapsack.dat knapsack.dat]
 * [source:pyomo.data.samples/trunk/pyomo/data/samples/scripts/s1/script.py script.py]

