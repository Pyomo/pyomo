.. _kernel_syntax_comparison:

Syntax Comparison Table (pyomo.kernel vs pyomo.environ)
=======================================================

.. list-table::
   :header-rows: 1
   :align: center

   * -
     - **pyomo.kernel**
     - **pyomo.environ**

   * - **Import**
     - .. literalinclude:: examples/kernel_example_Import_Syntax.spy
          :language: python
     - .. literalinclude:: examples/aml_example_Import_Syntax.spy
          :language: python
   * - **Model** [#models_fn]_
     - .. literalinclude:: examples/kernel_example_AbstractModels.spy
          :language: python
       .. literalinclude:: examples/kernel_example_ConcreteModels.spy
          :language: python
     - .. literalinclude:: examples/aml_example_AbstractModels.spy
          :language: python
       .. literalinclude:: examples/aml_example_ConcreteModels.spy
          :language: python
   * - **Set** [#sets_fn]_
     - .. literalinclude:: examples/kernel_example_Sets_1.spy
          :language: python
       .. literalinclude:: examples/kernel_example_Sets_2.spy
          :language: python
     - .. literalinclude:: examples/aml_example_Sets_1.spy
          :language: python
       .. literalinclude:: examples/aml_example_Sets_2.spy
          :language: python
   * - **Parameter** [#parameters_fn]_
     - .. literalinclude:: examples/kernel_example_Parameters_single.spy
          :language: python
       .. literalinclude:: examples/kernel_example_Parameters_dict.spy
          :language: python
       .. literalinclude:: examples/kernel_example_Parameters_list.spy
          :language: python
     - .. literalinclude:: examples/aml_example_Parameters_single.spy
          :language: python
       .. literalinclude:: examples/aml_example_Parameters_dict.spy
          :language: python
       .. literalinclude:: examples/aml_example_Parameters_list.spy
          :language: python
   * - **Variable**
     - .. literalinclude:: examples/kernel_example_Variables_single.spy
          :language: python
       .. literalinclude:: examples/kernel_example_Variables_dict.spy
          :language: python
       .. literalinclude:: examples/kernel_example_Variables_list.spy
          :language: python
     - .. literalinclude:: examples/aml_example_Variables_single.spy
          :language: python
       .. literalinclude:: examples/aml_example_Variables_dict.spy
          :language: python
       .. literalinclude:: examples/aml_example_Variables_list.spy
          :language: python
   * - **Constraint**
     - .. literalinclude:: examples/kernel_example_Constraints_single.spy
          :language: python
       .. literalinclude:: examples/kernel_example_Constraints_dict.spy
          :language: python
       .. literalinclude:: examples/kernel_example_Constraints_list.spy
          :language: python
     - .. literalinclude:: examples/aml_example_Constraints_single.spy
          :language: python
       .. literalinclude:: examples/aml_example_Constraints_dict.spy
          :language: python
       .. literalinclude:: examples/aml_example_Constraints_list.spy
          :language: python
   * - **Expression**
     - .. literalinclude:: examples/kernel_example_Expressions_single.spy
          :language: python
       .. literalinclude:: examples/kernel_example_Expressions_dict.spy
          :language: python
       .. literalinclude:: examples/kernel_example_Expressions_list.spy
          :language: python
     - .. literalinclude:: examples/aml_example_Expressions_single.spy
          :language: python
       .. literalinclude:: examples/aml_example_Expressions_dict.spy
          :language: python
       .. literalinclude:: examples/aml_example_Expressions_list.spy
          :language: python
   * - **Objective**
     - .. literalinclude:: examples/kernel_example_Objectives_single.spy
          :language: python
       .. literalinclude:: examples/kernel_example_Objectives_dict.spy
          :language: python
       .. literalinclude:: examples/kernel_example_Objectives_list.spy
          :language: python
     - .. literalinclude:: examples/aml_example_Objectives_single.spy
          :language: python
       .. literalinclude:: examples/aml_example_Objectives_dict.spy
          :language: python
       .. literalinclude:: examples/aml_example_Objectives_list.spy
          :language: python
   * - **SOS** [#sos_fn]_
     - .. literalinclude:: examples/kernel_example_SOS_single.spy
          :language: python
       .. literalinclude:: examples/kernel_example_SOS_dict.spy
          :language: python
       .. literalinclude:: examples/kernel_example_SOS_list.spy
          :language: python
     - .. literalinclude:: examples/aml_example_SOS_single.spy
          :language: python
       .. literalinclude:: examples/aml_example_SOS_dict.spy
          :language: python
       .. literalinclude:: examples/aml_example_SOS_list.spy
          :language: python
   * - **Suffix**
     - .. literalinclude:: examples/kernel_example_Suffix_single.spy
          :language: python
       .. literalinclude:: examples/kernel_example_Suffix_dict.spy
          :language: python
     - .. literalinclude:: examples/aml_example_Suffix_single.spy
          :language: python
       .. literalinclude:: examples/aml_example_Suffix_dict.spy
          :language: python
   * - **Piecewise** [#pw_fn]_
     - .. literalinclude:: examples/kernel_example_Piecewise_1d.spy
          :language: python
     - .. literalinclude:: examples/aml_example_Piecewise_1d.spy
          :language: python
.. [#models_fn] :python:`pyomo.kernel` does not include an alternative to the :python:`AbstractModel` component from :python:`pyomo.environ`. All data necessary to build a model must be imported by the user.
.. [#sets_fn] :python:`pyomo.kernel` does not include an alternative to the Pyomo :python:`Set` component from :python:`pyomo.environ`.
.. [#parameters_fn] :python:`pyomo.kernel.parameter` objects are always mutable.
.. [#sos_fn] Special Ordered Sets
.. [#pw_fn] Both :python:`pyomo.kernel.piecewise` and :python:`pyomo.kernel.piecewise_nd` create objects that are sub-classes of :python:`pyomo.kernel.block`. Thus, these objects can be stored in containers such as :python:`pyomo.kernel.block_dict` and :python:`pyomo.kernel.block_list`.
