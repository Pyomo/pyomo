Advanced Runtime Constraints
============================


Automated unpacking for multi-use black-boxes
---------------------------------------------

Coding a black-box model often represents a significant effort, and it is therefore desirable to have the black-box work in many situations.  A common case is to have a black-box model with a scalar input variable perform vectorized operations, ie, take in a vector and return a vector.  In the more general case, this can be thought of as passing in multiple run-cases as input to the black-box model.

The ``parseInputs()`` method enables this batch-like capability in a variety of forms:

.. py:function:: BlackBoxFunctionModel.parseInputs(self, *args, **kwargs)

    Parses the inputs to a black-box model into a list of run-cases

   :param args: The self attribute in all python methods
   :type  self: black-box model
   :param args: index passed arguments
   :type  args: list or tuple
   :param kwargs: keyword passed arguments
   :type  kwargs: dict
   
   :return: runcases
   :rtype: list
   :return: returnMode
   :rtype: int
   :return: extras
   :rtype:  list


The function definition is not particularly helpful, so let's dive in a bit.  For the typical user, we recommend that the top of all ``BlackBox()`` methods appear as follows:

.. code-block:: python

    def BlackBox(self, *args, **kwargs):
        runCases, returnMode, extras = self.parseInputs(*args, **kwargs)


Essentially, ``parseInputs()`` is a pre-processor that directly takes the inputs of the black-box.  The ``parseInputs()`` method will check all of the inputs, ensure that size and units are correct, split into run cases as appropriate, and return a run-cases list that is ready to operate on.

The ``runCases`` return (which can be named as any valid python name) is a list of dicts, where the keys to the dict are the names of the inputs declared in the ``__init__()`` method.  Ex: ``runCases[0]['x']`` will give the ``'x'`` input (in units specified in the ``__init__()`` method) in the first run-case.

The ``returnMode`` return (which can be named as any valid python name) is an integer that indicates how the return should be processed.  If ``returnMode`` is passed as a ``kwarg``, then the passed in value will be cast to this output.  If it is not passed in, then ``returnMode`` will be either ``-1*self.availableDerivative``, indicating that there is only a single run case, or ``self.availableDerivative`` which indicated multiple run cases.  A negative value for ``returnMode`` indicates that there will be one less layer of indexing in the output.

The ``extras`` return (which can be named as any valid python name) is a dict that will include all of the additional passed in values not expected by the black-box model.  The extra ``kwargs`` will appear as normal, and extra ``args`` get put in a list in ``extras['remainingArgs']``.  If you pass in a ``kwarg`` with key name ``'remainingArgs'``, it will be overwritten.  The extras dict is the place to find options that may affect the code (ex: tolerances, run modes, etc) that are not expected inputs to the black-box model.

Note that when using run case input, the output will always take the following unpacking structure:

::

    output[<run_case_index>][0] = <list_of_outputs_for_specified_runcase>
    output[<run_case_index>][0][<index_of_output>] = <output_for_specified_runcase>

    output[<run_case_index>][1] = <list_of_jacobians_for_specified_runcase>
    output[<run_case_index>][1][<index_of_output>][<index_of_input>] = <d(output_of_specified_index)/d(input_of_specified_index)>


There is **not** a shortcut for single scalar output black boxes as is the case when not using run cases, the final index to output 0 must be included.



An Example
++++++++++

There are many ways this functionality can be used, we provide an example here to get new users started


.. code-block:: python

    import numpy as np
    import pyomo.environ as pyo
    from pyomo.environ import units
    from pyomo.contrib.edi import Formulation, BlackBoxFunctionModel

    class SignomialTest(BlackBoxFunctionModel):
        def __init__(self):
            # Set up all the attributes by calling Model.__init__
            super().__init__()
            
            #Setup Inputs
            self.inputs.append( 'x', '', 'Independent Variable')
            
            #Setup Outputs
            self.outputs.append( 'y', '', 'Dependent Variable')
        
            #Simple model description
            self.description = ( 'This model evaluates the '+
                               'function: max([-6*x-6, x**4-3*x**2])' )
            
            self.availableDerivative = 1
        
        def BlackBox(self, *args, **kwargs):
            runCases, returnMode, extras = self.parseInputs(*args, **kwargs)

            x = np.array([ pyo.value(runCases[i]['x']) for i in range(0,len(runCases)) ])
            
            y = np.maximum(-6*x-6, x**4-3*x**2)
            dydx = 4*x**3 - 6*x
            ddy_ddx = 12*x**2 - 6
            gradientSwitch = -6*x-6 > x**4-3*x**2
            dydx[gradientSwitch] = -6
            ddy_ddx[gradientSwitch] = 0
        
            y = [ self.checkOutputs(yval) for yval in y ]
            dydx = [dydx[i] * units.dimensionless for i in range(0,len(dydx))]
            
            if returnMode < 0:
                returnMode = -1*(returnMode + 1)
                if returnMode == 0:
                    return y[0]
                if returnMode == 1:
                    return y[0], dydx
            else:
                if returnMode == 0:
                    opt = []
                    for i in range(0,len(y)):
                        opt.append([ y[i] ])
                    return opt
                if returnMode == 1:
                    opt = []
                    for i in range(0,len(y)):
                        opt.append([ [y[i]], [ [[dydx[i]]] ] ])
                    return opt

    s = SignomialTest()
    ivals = [[x] for x in np.linspace(-2,2,11)]

    # How the black box may be called using EDI
    bbo = s.BlackBox(**{'x':0.5})
    bbo = s.BlackBox({'x':0.5})
    bbo = s.BlackBox(**{'x':0.5, 'optn':True})

    # Additional options available with parseInputs
    bbo = s.BlackBox(*[0.5], **{'optn1': True, 'optn2': False})
    bbo = s.BlackBox(*[0.5,True], **{'optn': False})
    bbo = s.BlackBox({'x':[x for x in np.linspace(-2,2,11)]})
    bbo = s.BlackBox([{'x':x} for x in np.linspace(-2,2,11)])
    bbo = s.BlackBox([ [x] for x in np.linspace(-2,2,11)])
    bbo = s.BlackBox([ [x] for x in np.linspace(-2,2,11)], True, optn=False)
    bbo = s.BlackBox([ [x] for x in np.linspace(-2,2,11)], optn1=True, optn2=False)



Check outputs
-------------

There is a ``checkOutputs()`` method that is not supported in the current version.  Contact the developers if you desire this functionality, but the following the practices described in this documentation should render the need for this moot.


Cases of non-scalar inputs or outputs
-------------------------------------

Indexing can get somewhat complicated when inputs and outputs are not scalars.  Users should be warned this feature is supported, but not well tested, so please contact the developers if you encounter any unusual behavior.

The following unpacking remains the same:

::

    output[0] = <list_of_outputs>
    output[0][<index_of_output>] = <output>

    output[1] = <list_of_jacobians>
    output[1][<index_of_output>][<index_of_input>] = <d(output_of_specified_index)/d(input_of_specified_index)>

However, for outputs, the result will be an array with dimensions equal to the size of the output.  For Jacobians, it breaks down as the following:

::

    jacobian_list_entry[(output_dim_1_ix, output_dim_2_ix, ..., input_dim_1_ix, input_dim_2_ix, ...)] = <scalar_d(output_of_specified_index)/d(input_of_specified_index)>

For example, with an output that is ``NxN`` and an input that is also ``NxN``

::

    output[1][<index_of_output>][<index_of_input>][(0,0,1,1)] 

is the derivative of ``output[0,0]`` with respect to ``input[1,1]``



Tips
----

* A model summary can be printed by calling ``print(model_instance.summary)``









