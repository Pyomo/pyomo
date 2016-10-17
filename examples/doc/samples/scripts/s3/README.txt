= Working with OpenOpt Optimizers =

Pyomo integrates support for a variety of optimizers, including
optimizers that are supported by OpenOpt.  There are two ways that
OpenOpt optimizers can be used.  First, Pyomo models can be solve
using Pyomo's Python wrapper class for OpenOpt.  In this manner, OpenOpt 
optimizers look like any other Pyomo optimizer.  Second, a user can call a
utility function to generate a FuncDesigner oosolver object, which can be 
directly optimized.

== Example 1 ==

The `pyomo` command includes the `--solver` option, which can be
used to specify OpenOpt optimizers with the `openopt:` prefix:
{{{
  pyomo --solver=openopt:ralg pyomo_nlp1.py
}}}
OpenOpt includes interface to a wide range of optimizers, many of which are supported
through third-party interfaces.  For example, if the CVXOPT package is installed, then
OpenOpt includes an interface to the GLPK mixed-integer linear programming solver:
{{{
  pyomo --solver=openopt:glpk lp1.py
}}}


== Example 2 ==

OpenOpt users can also leverage Pyomo's modeling functionality by explicitly converting 
a Pyomo model into a FuncDesigner `oosystem` object.  For example, the following script is adapted from the 
`nlp1` example that is provided with OpenOpt distributions:[[BR]]
[[Include(source:pyomo.data.samples/trunk/pyomo/data/samples/scripts/s3/openopt_nlp1.py)]]
The following script creates a concrete Pyomo model, creates the oosystem object, and then performs
optimization with an OpenOpt optimizer:
[[Include(source:pyomo.data.samples/trunk/pyomo/data/samples/scripts/s3/pyomo_nlp1.py)]]

== Downloads ==

 * [source:pyomo.data.samples/trunk/pyomo/data/samples/scripts/s2/openopt_nlp1.py openopt_nlp1.py]
 * [source:pyomo.data.samples/trunk/pyomo/data/samples/scripts/s2/pyomo_nlp1.py pyomo_nlp1.py]
 * [source:pyomo.data.samples/trunk/pyomo/data/samples/scripts/s2/nlp1.py nlp1.py]
 * [source:pyomo.data.samples/trunk/pyomo/data/samples/scripts/s2/lp1.py lp1.py]

