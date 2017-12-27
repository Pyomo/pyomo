Managing Data in Pyomo Models
=============================

The :class:`Set <ref pyomo.core.base.set.Set>` and :class:`Param <ref pyomo.core.base.param.Param>` components of a Pyomo model are
used to define data values used to construct constraints and
objectives.  Previous chapters have illustrated that these components
are not necessary to develop complex models.  However, The :class:`Set <ref pyomo.core.base.set.Set>` and :class:`Param <ref pyomo.core.base.param.Param>`
components can be used to define abstract data
declarations, where no data values are specified.  For example::

    model.A = Set(within=Reals)
    model.p = Param(model.A, within=Integers)

Data command files can be used to initialize data declarations in
Pyomo models, and in particular they are useful for initializing
abstract data declarations.


.. toctree::
   :maxdepth: 1

   datfiles.rst
   dataportals.rst
   other.rst
