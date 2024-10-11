Model Scaling Transformation
============================

Good scaling of models can greatly improve the numerical properties of a problem and thus increase reliability and convergence. The ``core.scale_model`` transformation allows users to separate scaling of a model from the declaration of the model variables and constraints which allows for models to be written in more natural forms and to be scaled and rescaled as required without having to rewrite the model code.

.. autosummary::

   pyomo.core.plugins.transform.scaling.ScaleModel


Setting Scaling Factors
-----------------------

Scaling factors for components in a model are declared using :ref:`Suffixes`, as shown in the example above. In order to define a scaling factor for a component, a ``Suffix`` named ``scaling_factor`` must first be created to hold the scaling factor(s). Scaling factor suffixes can be declared at any level of the model hierarchy, but scaling factors declared on the higher-level ``models`` or ``Blocks`` take precedence over those declared at lower levels.

Scaling suffixes are dict-like where each key is a Pyomo component and the value is the scaling factor to be applied to that component.

In the case of indexed components, scaling factors can either be declared for an individual index or for the indexed component as a whole (with scaling factors for individual indices taking precedence over overall scaling factors).

.. note::

   In the case that a scaling factor is declared for a component on at multiple levels of the hierarchy, the highest level scaling factor will be applied.

.. note::

   It is also possible (but not encouraged) to define a "default" scaling factor to be applied to any component for which a specific scaling factor has not been declared by setting a entry in a Suffix with a key of ``None``. In this case, the default value declared closest to the component to be scaled will be used (i.e., the first default value found when walking up the model hierarchy).

Applying Model Scaling
----------------------

The ``core.scale_model`` transformation provides two approaches for creating a scaled model.

In-Place Scaling
****************

The ``apply_to(model)`` method can be used to apply scaling directly to an existing model. When using this method, all the variables, constraints and objectives within the target model are replaced with new scaled components and the appropriate scaling factors applied. The model can then be sent to a solver as usual, however the results will be in terms of the scaled components and must be un-scaled by the user.

Creating a New Scaled Model
***************************

Alternatively, the ``create_using(model)`` method can be used to create a new, scaled version of the model which can be solved. In this case, a clone of the original model is generated with the variables, constraints and objectives replaced by scaled equivalents. Users can then send the scaled model to a solver after which the ``propagate_solution`` method can be used to map the scaled solution back onto the original model for further analysis.

The advantage of this approach is that the original model is maintained separately from the scaled model, which facilitates rescaling and other manipulation of the original model after a solution has been found. The disadvantage of this approach is that cloning the model may result in memory issues when dealing with larger models.
