Stochastic Programming
======================

To express a stochastic program in PySP, the user specifies both the
deterministic base model and the scenario tree model with associated
uncertain parameters. Both concrete and abstract model representations
are supported.

Given the deterministic and scenario tree models, PySP provides
multiple paths for the solution of the corresponding stochastic
program. One alternative involves forming the extensive form and
invoking an appropriate deterministic solver for the entire problem
once. For more complex stochastic programs, we provide a generic
implementation of Rockafellar and Wets' Progressive Hedging algorithm,
with additional specializations for approximating mixed-integer
stochastic programs as well as other decomposition methods. By
leveraging the combination of a high-level programming language
(Python) and the embedding of the base deterministic model in that
language (Pyomo), we are able to provide completely generic and highly
configurable solver implementations.

See the Pysp section :ref:`pysp-overview` in Getting Started for more information.
