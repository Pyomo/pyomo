Constraints
===========









Known Issues
------------

* Currently only equality constraints are supported, pending an update to pyomo (see `this issue <https://github.com/codykarcher/pyomo/issues/2>`_)
* Indexed variables must be broken up using either indicies or a pyomo rule (see `this issue <https://github.com/codykarcher/pyomo/issues/3>`_)
* Units that are inconsistent, but not the same (ie, meters and feet) will flag as invalid when checking units (see `this issue <https://github.com/codykarcher/pyomo/issues/6>`_)
