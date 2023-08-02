#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2023
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#
#  Development of this module was conducted as part of the Institute for
#  the Design of Advanced Energy Systems (IDAES) with support through the
#  Simulation-Based Engineering, Crosscutting Research Program within the
#  U.S. Department of Energy’s Office of Fossil Energy and Carbon Management.
#
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import copy
import numpy as np
import pyomo
import pyomo.environ as pyo
from pyomo.environ import units as pyomo_units
import scipy.sparse as sps
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel, ExternalGreyBoxBlock


class BlackBoxFunctionModel_Variable(object):
    def __init__(self, name, units, description = '', size = 0):
        # Order matters
        self.name = name
        self.units = units
        self.size = size
        self.description = description
# =====================================================================================================================
# The printing function
# =====================================================================================================================
    def __repr__(self):
        return self.name    
# =====================================================================================================================
# Define the name
# =====================================================================================================================
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self,val):
        if isinstance(val, str):
            self._name = val
        else:
            raise ValueError('Invalid name.  Must be a string.')
# =====================================================================================================================
# Define the units
# =====================================================================================================================
    @property
    def units(self):
        return self._units
    @units.setter
    def units(self, val):
        # set dimensionless if a null string is passed in
        if isinstance(val, str):
            if val in ['-', '', ' ']:
                val = 'dimensionless'
        
        if isinstance(val, str):
            self._units = pyomo_units.__getattr__(val)
        elif isinstance(val, pyomo.core.base.units_container._PyomoUnit):
            self._units = val
        else:
            raise ValueError('Invalid units.  Must be a string compatible with pint or a unit instance.')
# =====================================================================================================================
# Define the size
# =====================================================================================================================
    @property
    def size(self):
        return self._size
    @size.setter
    def size(self, val):
        invalid = False
        if isinstance(val,(list, tuple)):
            sizeTemp = []
            for x in val:
                if isinstance(x, str):
                    # is a vector of unknown length, should be 'inf', but any string accepted
                    x = -1
                    # pass
                elif not isinstance(x,int):
                    raise ValueError('Invalid size.  Must be an integer or list/tuple of integers')
                if x == 1:
                    raise ValueError('A value of 1 is not valid for defining size.  Use fewer dimensions.')
                sizeTemp.append(x)
            self._size = val
        else:
            if val is None:
                self._size = None
            elif isinstance(val, str):
                # is a 1D vector of unknown length, should be 'inf', but any string accepted
                self._size = -1
                # pass
            elif isinstance(val, int):
                if val == 1:
                    raise ValueError('A value of 1 is not valid for defining size.  Use 0 to indicate a scalar value.')
                else:
                    self._size = val
            else:
                raise ValueError('Invalid size.  Must be an integer or list/tuple of integers')
# =====================================================================================================================
# Define the description
# =====================================================================================================================
    @property
    def description(self):
        return self._description
    @description.setter
    def description(self, val):
        if isinstance(val,str):
            self._description = val
        else:
            raise ValueError('Invalid description.  Must be a string.')





class TypeCheckedList(list):
    def __init__(self, checkItem, itemList = None):
        super(TypeCheckedList, self).__init__()
        self.checkItem = checkItem
        
        if itemList is not None:
            if isinstance(itemList, list) or isinstance(itemList, tuple):
                for itm in itemList:
                    self.append(itm)
            else:
                raise ValueError('Input to itemList is not iterable')
        
    def __setitem__(self, key, val):
        if isinstance(val, self.checkItem):
            super(TypeCheckedList, self).__setitem__(key, val)
        else:
            raise ValueError('Input must be an instance of the defined type')
        
    def __setslice__(self, i, j, sequence):
        performSet = True
        for val in sequence:
            if not isinstance(val, self.checkItem):
                performSet = False
                break
        
        if performSet:
            super(TypeCheckedList, self).__setslice__(i,j,sequence)
        else:
            raise ValueError('All values in the input must be an instance of the defined type')
        
    def append(self, val):
        if isinstance(val, self.checkItem):
            super(TypeCheckedList, self).append(val)
        else:
            raise ValueError('Input must be an instance of the defined type')


class BBList(TypeCheckedList):
    def __init__(self):
        super(BBList, self).__init__(BlackBoxFunctionModel_Variable,[])
        self._lookupDict = {}
        self._counter = 0

    def __getitem__(self, val):
        if isinstance(val, int):
            return super(BBList, self).__getitem__(val)
        elif isinstance(val, str):
            return super(BBList, self).__getitem__(self._lookupDict[val])
        else:
            raise ValueError('Input must be an integer or a valid variable name')



    def append(*args, **kwargs):
        args = list(args)
        self = args.pop(0)
        
        if len(args) + len(kwargs.values()) == 1:
            if len(args) == 1:
                inputData = args[0]
            if len(kwargs.values()) == 1:
                inputData = list(kwargs.values())[0]
                
            if isinstance(inputData, self.checkItem):
                if inputData.name in self._lookupDict.keys():
                    raise ValueError("Key '%s' already exists in the input list"%(inputData.name))
                self._lookupDict[inputData.name] = self._counter
                self._counter += 1
                super(BBList, self).append(inputData)
            else:
                if isinstance(inputData, str):
                    raise ValueError("Key '%s' not passed in to the black box variable constructor"%('units'))
                else:
                    raise ValueError('Invalid (single) input type')
        
        elif len(args) + len(kwargs.values()) <= 4:
            argKeys = ['name','units','description','size']
            ipd = dict(zip(argKeys[0:len(args)],args))
            for ky, vl in kwargs.items():
                if ky in ipd:
                    raise ValueError("Key '%s' declared after non-keyword arguments and is out of order"%(ky))
                else:
                    ipd[ky]=vl
            
            for ak in argKeys:
                if ak not in ipd.keys():
                    if ak == 'description':
                        ipd['description']=''
                    elif ak == 'size':
                        ipd['size'] = 0
                    else:
                        raise ValueError("Key '%s' not passed in to the black box variable constructor"%(ak))
                        

            if ipd['name'] in self._lookupDict.keys():
                raise ValueError("Key '%s' already exists in the input list"%(ipd['name']))
            self._lookupDict[ipd['name']] = self._counter
            self._counter += 1
            super(BBList, self).append(BlackBoxFunctionModel_Variable(**ipd))
            
        else:
            raise ValueError('Too many inputs to a black box variable')


errorString = 'This function is calling to the base class and has not been defined.'

class BlackBoxFunctionModel(ExternalGreyBoxModel):
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super(BlackBoxFunctionModel, self).__init__()

        # List of the inputs and outputs
        self.inputs = BBList()
        self.outputs = BBList()

        self.inputVariables_optimization  = None
        self.outputVariables_optimization = None
        
        # A simple description of the model
        self.description = None

        # Defines the order of derivative available in the black box
        self.availableDerivative = 0

        self._cache = None
        self._NunwrappedOutputs = None
        self._NunwrappedInputs  = None

    def setOptimizationVariables(self, inputVariables_optimization, outputVariables_optimization):
        self.inputVariables_optimization  = inputVariables_optimization
        self.outputVariables_optimization = outputVariables_optimization

# ---------------------------------------------------------------------------------------------------------------------
# pyomo things
# ---------------------------------------------------------------------------------------------------------------------
    def input_names(self):
        inputs_unwrapped = []
        for ivar in self.inputVariables_optimization:
            if isinstance(ivar, pyomo.core.base.var.ScalarVar):
                inputs_unwrapped.append(ivar)
            elif isinstance(ivar, pyomo.core.base.var.IndexedVar):
                validIndicies = list(ivar.index_set().data())
                for vi in validIndicies:
                    inputs_unwrapped.append(ivar[vi])
            else:
                print(type(ivar))
                raise ValueError("Invalid type for input variable") 

        return [ip.__str__() for ip in  inputs_unwrapped]


    def output_names(self):
        outputs_unwrapped = []
        for ovar in self.outputVariables_optimization:
            if isinstance(ovar, pyomo.core.base.var.ScalarVar):
                outputs_unwrapped.append(ovar)
            elif isinstance(ovar, pyomo.core.base.var.IndexedVar):
                validIndicies = list(ovar.index_set().data())
                for vi in validIndicies:
                    outputs_unwrapped.append(ovar[vi])
            else:
               raise ValueError("Invalid type for output variable") 

        return [op.__str__() for op in outputs_unwrapped]

    def set_input_values(self, input_values):
        self._input_values = input_values
        self._cache = None

    def evaluate_outputs(self):
        self.fillCache()
        opts = self._cache['pyomo_outputs']
        return opts
        

    def evaluate_jacobian_outputs(self):
        self.fillCache()
        jac = self._cache['pyomo_jacobian']
        return jac

    def post_init_setup(self, defaultVal = 1.0):
        self._input_values = np.ones(self._NunwrappedInputs) * defaultVal


    def fillCache(self):
        if self._cache is None:
            self._cache = {}

            raw_inputs = self._input_values
            bb_inputs = []

            ptr = 0

            for i in range(0,len(self.inputVariables_optimization)):
                optimizationInput = self.inputVariables_optimization[i]
                ipt               = self.inputs[i]

                shape             = [len(idx) for idx in optimizationInput.index_set().subsets()]
                localShape        = ipt.size

                optimizationUnits = self.inputVariables_optimization[i].get_units()
                localUnits        = ipt.units

                if isinstance(optimizationInput, pyomo.core.base.var.IndexedVar):
                    value = np.zeros(shape) 
                    for vix in list(optimizationInput.index_set().data()):
                        raw_val                = float(raw_inputs[ptr]) * optimizationUnits
                        raw_val_correctedUnits = pyomo_units.convert(raw_val, localUnits)
                        value[vix] = pyo.value(raw_val_correctedUnits)
                        ptr += 1
                    self.sizeCheck(localShape, value*localUnits)
                    bb_inputs.append(value*localUnits)

                elif isinstance(optimizationInput, pyomo.core.base.var.ScalarVar):
                    value = raw_inputs[ptr] * optimizationUnits
                    ptr += 1
                    self.sizeCheck(localShape, value)
                    value_correctedUnits = pyomo_units.convert(value, localUnits)
                    bb_inputs.append(value_correctedUnits)

                else:
                    raise ValueError("Invalid input variable type")

            bbo = self.BlackBox(*bb_inputs)

            self._cache['raw']          = bbo
            self._cache['raw_value']    = bbo[0] 
            self._cache['raw_jacobian'] = bbo[1] 

            outputVector = []
            if not isinstance(bbo[0], (list,tuple)):
                valueList = [bbo[0]]
                jacobianList = [bbo[1]]
            else:
                valueList = bbo[0]
                jacobianList = bbo[1]
            for i in range(0,len(valueList)):
                optimizationOutput = self.outputVariables_optimization[i]
                opt                = self.outputs[i]

                modelOutputUnits        = opt.units
                ouptutOptimizationUnits = optimizationOutput.get_units()
                vl = valueList[i]
                if isinstance(vl, pyomo.core.expr.numeric_expr.NumericNDArray):
                    validIndexList = optimizationOutput.index_set().data()
                    for j in range(0,len(validIndexList)):
                        vi = validIndexList[j]
                        corrected_value = pyo.value(pyomo_units.convert(vl[vi], ouptutOptimizationUnits)) # now unitless in correct units
                        outputVector.append(corrected_value) 

                elif isinstance(vl, (pyomo.core.expr.numeric_expr.NPV_ProductExpression,
                                  pyomo.core.base.units_container._PyomoUnit)):
                    corrected_value = pyo.value(pyomo_units.convert(vl, ouptutOptimizationUnits)) # now unitless in correct units
                    outputVector.append(corrected_value) 

                else:
                    raise ValueError("Invalid output variable type")

            self._cache['pyomo_outputs'] = outputVector

            outputJacobian = np.ones([self._NunwrappedOutputs, self._NunwrappedInputs]) * -1
            ptr_row = 0
            ptr_col = 0 

            for i in range(0,len(jacobianList)):
                oopt = self.outputVariables_optimization[i]
                lopt = self.outputs[i]
                oounits = oopt.get_units()
                lounits = lopt.units
                # oshape  = [len(idx) for idx in oopt.index_set().subsets()]
                ptr_col = 0
                for j in range(0,len(self.inputs)):
                    oipt = self.inputVariables_optimization[j]
                    lipt = self.inputs[j]
                    oiunits = oipt.get_units()
                    liunits = lipt.units
                    # ishape  = [len(idx) for idx in oipt.index_set().subsets()]

                    jacobianValue_raw = jacobianList[i][j]

                    if isinstance(jacobianValue_raw, (pyomo.core.expr.numeric_expr.NPV_ProductExpression,
                                                      pyomo.core.base.units_container._PyomoUnit)):
                        corrected_value = pyo.value(pyomo_units.convert(jacobianValue_raw, oounits/oiunits)) # now unitless in correct units
                        outputJacobian[ptr_row,ptr_col] = corrected_value
                        ptr_col += 1
                        ptr_row_step = 1

                    elif isinstance(jacobianValue_raw,pyomo.core.expr.numeric_expr.NumericNDArray):
                        jshape = jacobianValue_raw.shape

                        if isinstance(oopt, pyomo.core.base.var.ScalarVar):
                            oshape = 0
                        elif isinstance(oopt, pyomo.core.base.var.IndexedVar):
                            oshape  = [len(idx) for idx in oopt.index_set().subsets()]
                        else:
                            raise ValueError("Invalid type for output variable") 

                        if isinstance(oipt, pyomo.core.base.var.ScalarVar):
                            ishape = 0
                        elif isinstance(oipt, pyomo.core.base.var.IndexedVar):
                            ishape  = [len(idx) for idx in oipt.index_set().subsets()]
                        else:
                            raise ValueError("Invalid type for input variable") 

                        if oshape == 0:
                            validIndicies = list(oipt.index_set().data())
                            for vi in validIndicies:
                                corrected_value = pyo.value(pyomo_units.convert(jacobianValue_raw[vi], oounits/oiunits)) # now unitless in correct units
                                outputJacobian[ptr_row,ptr_col] = corrected_value
                                ptr_col += 1
                            ptr_row_step = 1

                        elif ishape == 0:
                            ptr_row_cache = ptr_row
                            validIndicies = list(oopt.index_set().data())
                            for vi in validIndicies:
                                corrected_value = pyo.value(pyomo_units.convert(jacobianValue_raw[vi], oounits/oiunits)) # now unitless in correct units
                                outputJacobian[ptr_row,ptr_col] = corrected_value
                                ptr_row += 1
                            ptr_row = ptr_row_cache
                            ptr_row_step = len(validIndicies)

                        # elif ishape == 0 and oshape == 0: # Handled by the scalar case above

                        else: 
                            # both are dimensioned vectors
                            #oshape, ishape, jshape
                            ptr_row_cache = ptr_row
                            ptr_col_cache = ptr_col
                            validIndicies_o = list(oopt.index_set().data())
                            validIndicies_i = list(oipt.index_set().data())

                            for vio in validIndicies_o:
                                if isinstance(vio, (float,int)):
                                    vio = (vio,)
                                for vii in validIndicies_i:
                                    if isinstance(vii, (float,int)):
                                        vii = (vii,)
                                    corrected_value = pyo.value(pyomo_units.convert(jacobianValue_raw[vio + vii], oounits/oiunits)) # now unitless in correct units
                                    outputJacobian[ptr_row,ptr_col] = corrected_value
                                    ptr_col += 1
                                ptr_col = ptr_col_cache
                                ptr_row += 1
                            ptr_row = ptr_row_cache
                            ptr_row_step = len(validIndicies_o)

                    else:
                        raise ValueError("Invalid jacobian type")

                ptr_row += ptr_row_step 

            self._cache['pyomo_jacobian'] = sps.coo_matrix(outputJacobian)

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
    # These models must be defined in each individual model, just placeholders here
    def BlackBox(*args, **kwargs):
        raise AttributeError(errorString)

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
    def parseInputs(*args, **kwargs):
        args = list(args)       # convert tuple to list
        self = args.pop(0)      # pop off the self argument

        inputNames = [self.inputs[i].name for i in range(0,len(self.inputs))]

        # ------------------------------
        # ------------------------------
        if len(args) + len(kwargs.values()) == 1:
            if len(args) == 1:
                inputData = args[0]
            if len(kwargs.values()) == 1:
                inputData = list(kwargs.values())[0]

            if len(inputNames) == 1:
                try:
                    rs = self.sanitizeInputs(inputData)
                    return [dict(zip(inputNames,[rs]))], -self.availableDerivative-1, {} # one input being passed in
                except:
                    pass #otherwise, proceed
        
            if isinstance(inputData, (list,tuple)):
                dataRuns = []
                for idc in inputData:
                    if isinstance(idc, dict):
                        sips = self.sanitizeInputs(**idc)
                        if len(inputNames) == 1:
                            sips = [sips]
                        runDictS = dict(zip(inputNames,sips))
                        dataRuns.append(runDictS)                # the BlackBox([{'x1':x1, 'x2':x2},{'x1':x1, 'x2':x2},...]) case
                    elif isinstance(idc,(list,tuple)):
                        if len(idc) == len(inputNames):
                            sips = self.sanitizeInputs(*idc)
                            if len(inputNames) == 1:
                                sips = [sips]
                            runDictS = dict(zip(inputNames,sips))
                            dataRuns.append(runDictS)                # the BlackBox([ [x1, x2], [x1, x2],...]) case
                        else:
                            raise ValueError('Entry in input data list has improper length')
                    else:
                        raise ValueError("Invalid data type in the input list.  Note that BlackBox([x1,x2,y]) must be passed in as BlackBox([[x1,x2,y]]) or "+
                                                "BlackBox(*[x1,x2,y]) or BlackBox({'x1':x1,'x2':x2,'y':y}) or simply BlackBox(x1, x2, y) to avoid processing singularities.  "+ 
                                                "Best practice is BlackBox({'x1':x1,'x2':x2,'y':y})")
                return dataRuns, self.availableDerivative, {}

            elif isinstance(inputData, dict):
                if set(list(inputData.keys())) == set(inputNames):
                    try:
                        inputLengths = [len(inputData[kw]) for kw in inputData.keys()]
                    except:
                        sips = self.sanitizeInputs(**inputData)
                        if len(inputNames) == 1:
                            sips = [sips]
                        return [dict(zip(inputNames,sips))], -self.availableDerivative-1, {} # the BlackBox(*{'x1':x1, 'x2':x2}) case, somewhat likely

                    if not all([inputLengths[i] == inputLengths[0] for i in range(0,len(inputLengths))]):
                        sips = self.sanitizeInputs(**inputData)
                        return [dict(zip(inputNames,sips))], -self.availableDerivative-1, {} # the BlackBox(*{'x1':x1, 'x2':x2}) case where vectors for x1... are passed in (likely to fail on previous line)
                    else:
                        try:
                            sips = self.sanitizeInputs(**inputData)
                            return [dict(zip(inputNames,sips))], -self.availableDerivative-1, {} # the BlackBox(*{'x1':x1, 'x2':x2}) case where vectors all inputs have same length intentionally (likely to fail on previous line)
                        except:
                            dataRuns = []
                            for i in range(0,inputLengths[0]):
                                runDict = {}
                                for ky,vl in inputData.items():
                                    runDict[ky] = vl[i]
                                sips = self.sanitizeInputs(**runDict)
                                if len(inputNames) == 1:
                                    sips = [sips]
                                runDictS = dict(zip(inputNames,sips))
                                dataRuns.append(runDictS)
                            return dataRuns, self.availableDerivative, {} # the BlackBox({'x1':x1_vec, 'x2':x2_vec}) case, most likely
                else:
                    raise ValueError('Keywords did not match the exptected list')
            else:
                raise ValueError('Got unexpected data type %s'%(str(type(inputData))))
        # ------------------------------
        # ------------------------------
        else:
            if any([ list(kwargs.keys())[i] in inputNames for i in range(0,len(list(kwargs.keys()))) ]):
                # some of the inputs are defined in the kwargs
                if len(args) >= len(inputNames):
                    raise ValueError('A keyword input is defining an input, but there are too many unkeyed arguments for this to occour.  Check the inputs.')
                else:
                    if len(args) != 0:
                        availableKeywords = inputNames[-len(args):]
                    else: 
                        availableKeywords = inputNames

                    valList = args + [None]*(len(inputNames)-len(args))
                    for ky in availableKeywords:
                        ix = inputNames.index(ky)
                        valList[ix] = kwargs[ky]

                    if any([valList[i]==None for i in range(0,len(valList))]):
                        raise ValueError('Kewords did not properly fill in the remaining arguments. Check the inputs.')

                    sips = self.sanitizeInputs(*valList)
                    if len(inputNames) == 1:
                        sips = [sips]

                    remainingKwargs = copy.deepcopy(kwargs)
                    for nm in inputNames:
                        del remainingKwargs[nm]

                    return [dict(zip(inputNames,sips))], -self.availableDerivative-1, remainingKwargs # Mix of args and kwargs define inputs
            else:
                # all of the inputs are in the args
                try:
                    sips = self.sanitizeInputs(*args[0:len(inputNames)])
                    if len(inputNames) == 1:
                        sips = [sips]

                    remainingKwargs = copy.deepcopy(kwargs)
                    remainingKwargs['remainingArgs'] = args[len(inputNames):]
                    return [dict(zip(inputNames,sips))], -self.availableDerivative-1, remainingKwargs # all inputs are in args
                except:
                    runCases, returnMode, extra_singleInput = self.parseInputs(args[0])
                    remainingKwargs = copy.deepcopy(kwargs)
                    remainingKwargs['remainingArgs'] = args[len(inputNames):]
                    return runCases, returnMode, remainingKwargs

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
    def sizeCheck(self, size, ipval_correctUnits):
        if size is not None:
            szVal = ipval_correctUnits
            if isinstance(szVal, (pyomo.core.expr.numeric_expr.NPV_ProductExpression,
                                  pyomo.core.base.units_container._PyomoUnit)):
                if size != 0 and size != 1 :
                    raise ValueError('Size of %s did not match the expected size %s (ie: Scalar)'%(name, str(size)))
            elif isinstance(szVal, pyomo.core.expr.numeric_expr.NumericNDArray):
                shp = szVal.shape
                if isinstance(size,(int,float)):
                    size = [size]
                # else:
                if len(shp) != len(size):
                    raise ValueError('Shapes/Sizes of %s does not match the expected %s'%(str(shp),str(size)))
                for j in range(0,len(shp)):
                    if size[j] != -1:  # was declared of flexible length
                        if size[j] != shp[j]:
                            raise ValueError('Shapes/Sizes of %s does not match the expected %s'%(str(shp),str(size)))
            else:
                raise ValueError('Invalid type detected when checking size (Should never display)')

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
    def sanitizeInputs(self, *args, **kwargs):
        nameList = [self.inputs[i].name for i in range(0,len(self.inputs))]
        if len(args) + len(kwargs.values()) > len(nameList):
            raise ValueError('Too many inputs')
        if len(args) + len(kwargs.values()) < len(nameList):
            raise ValueError('Not enough inputs')
        inputDict = {}
        for i in range(0,len(args)):
            rg = args[i]
            inputDict[nameList[i]] = rg

        for ky, vl in kwargs.items():
            if ky in nameList:
                inputDict[ky] = vl
            else:
                raise ValueError('Unexpected input keyword argument %s in the inputs'%(ky))

        opts = []

        for i in range(0,len(nameList)):
            name = nameList[i]
            nameCheck = self.inputs[i].name
            unts = self.inputs[i].units
            size = self.inputs[i].size
            
            if name != nameCheck:
                raise RuntimeError('Something went wrong and values are not consistent.  Check your defined inputs.')

            ipval = inputDict[name]

            if unts is not None:
                if isinstance(ipval,  pyomo.core.expr.numeric_expr.NumericNDArray):
                    for ii in range(0,len(ipval)):
                        try:
                            ipval[ii] = pyomo_units.convert(ipval[ii], unts)#ipval.to(unts)
                        except:
                            raise ValueError('Could not convert %s of %s to %s'%(name, str(ipval),str(unts)))
                    ipval_correctUnits = ipval
                else:
                    try:
                        ipval_correctUnits = pyomo_units.convert(ipval, unts)#ipval.to(unts)
                    except:
                        raise ValueError('Could not convert %s of %s to %s'%(name, str(ipval),str(unts)))
            else:
                ipval_correctUnits = ipval

            if not isinstance(ipval_correctUnits, (pyomo.core.expr.numeric_expr.NPV_ProductExpression,
                                                   pyomo.core.expr.numeric_expr.NumericNDArray,
                                                   pyomo.core.base.units_container._PyomoUnit)):
                ipval_correctUnits = ipval_correctUnits * pyomo_units.dimensionless

            self.sizeCheck(size, ipval_correctUnits)

            opts.append(ipval_correctUnits)
        if len(opts) == 1:
            opts = opts[0]

        return opts

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
    def checkOutputs(self, *args, **kwargs):
        nameList = [self.outputs[i].name for i in range(0,len(self.outputs))]
        if len(args) + len(kwargs.values()) > len(nameList):
            raise ValueError('Too many inputs')
        if len(args) + len(kwargs.values()) < len(nameList):
            raise ValueError('Not enough inputs')
        inputDict = {}
        for i in range(0,len(args)):
            rg = args[i]
            inputDict[nameList[i]] = rg

        for ky, vl in kwargs.items():
            if ky in nameList:
                inputDict[ky] = vl
            else:
                raise ValueError('Unexpected input keyword argument %s in the inputs'%(ky))

        opts = []

        for i in range(0,len(nameList)):
            name = nameList[i]
            nameCheck = self.outputs[i].name
            unts = self.outputs[i].units
            size = self.outputs[i].size
            
            if name != nameCheck:
                raise RuntimeError('Something went wrong and values are not consistent.  Check your defined inputs.')

            ipval = inputDict[name]

            if unts is not None:
                try:
                    ipval_correctUnits = pyomo_units.convert(ipval, unts)
                except:
                    raise ValueError('Could not convert %s of %s to %s'%(name, str(ipval),str(unts)))
            else:
                ipval_correctUnits = ipval

            if not isinstance(ipval_correctUnits, (pyomo.core.expr.numeric_expr.NPV_ProductExpression,
                                                   pyomo.core.expr.numeric_expr.NumericNDArray,
                                                   pyomo.core.base.units_container._PyomoUnit)):
                ipval_correctUnits = ipval_correctUnits * pyomo_units.dimensionless

            self.sizeCheck(size, ipval_correctUnits)

            opts.append(ipval_correctUnits)
        if len(opts) == 1:
            opts = opts[0]

        return opts

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
    def getSummary(self, whitespace = 6):
        pstr = '\n'
        pstr += 'Model Description\n'
        pstr += '=================\n'
        descr_str = self.description.__repr__()
        pstr += descr_str[1:-1] + '\n\n'
        
        longestName = 0
        longestUnits = 0
        longestSize = 0
        for ipt in self.inputs:
            nml = len(ipt.name)
            if nml > longestName:
                longestName = nml
            if ipt.units is None:
                unts = 'None'
            else:
                unts = ipt.units.__str__()#_repr_html_()
                # unts = unts.replace('<sup>','^')
                # unts = unts.replace('</sup>','')
                # unts = unts.replace('\[', '[')
                # unts = unts.replace('\]', ']')
            unl = len(unts)
            if unl > longestUnits:
                longestUnits = unl
            if ipt.size is None:
                lsz = 4
            else:
                if type(ipt.size) == list:
                    lsz = len(ipt.size.__repr__())
                else:
                    lsz = len(str(ipt.size))
            if lsz > longestSize:
                longestSize = lsz
        namespace = max([4, longestName]) + whitespace
        unitspace = max([5, longestUnits]) + whitespace
        sizespace = max([4, longestSize]) + whitespace
        fulllength = namespace+unitspace+sizespace+11
        pstr += 'Inputs\n'
        pstr += '='*fulllength
        pstr += '\n'
        pstr += 'Name'.ljust(namespace)
        pstr += 'Units'.ljust(unitspace)
        pstr += 'Size'.ljust(sizespace)
        pstr += 'Description'
        pstr += '\n'
        pstr += '-'*(namespace - whitespace)
        pstr += ' '*whitespace
        pstr += '-'*(unitspace - whitespace)
        pstr += ' '*whitespace
        pstr += '-'*(sizespace - whitespace)
        pstr += ' '*whitespace
        pstr += '-----------'
        pstr += '\n'
        for ipt in self.inputs:
            pstr += ipt.name.ljust(namespace)
            if ipt.units is None:
                unts = 'None'
            else:
                unts = ipt.units.__str__()#_repr_html_()
                # unts = unts.replace('<sup>','^')
                # unts = unts.replace('</sup>','')
                # unts = unts.replace('\[', '[')
                # unts = unts.replace('\]', ']')
            pstr += unts.ljust(unitspace)
            if ipt.size is None:
                lnstr = 'None'
            else:
                lnstr = '%s'%(ipt.size.__repr__())
            pstr += lnstr.ljust(sizespace)
            pstr += ipt.description
            pstr += '\n'
        pstr += '\n'
        
        longestName = 0
        longestUnits = 0
        longestSize = 0
        for opt in self.outputs:
            nml = len(opt.name)
            if nml > longestName:
                longestName = nml
            if opt.units is None:
                unts = 'None'
            else:
                unts = opt.units.__str__()#_repr_html_()
                # unts = unts.replace('<sup>','^')
                # unts = unts.replace('</sup>','')
                # unts = unts.replace('\[', '[')
                # unts = unts.replace('\]', ']')
            unl = len(unts)
            if unl > longestUnits:
                longestUnits = unl
            if opt.size is None:
                lsz = 4
            else:
                if type(opt.size) == list:
                    lsz = len(opt.size.__repr__())
                else:
                    lsz = len(str(opt.size))
            if lsz > longestSize:
                longestSize = lsz
        namespace = max([4, longestName]) + whitespace
        unitspace = max([5, longestUnits]) + whitespace
        sizespace = max([4, longestSize]) + whitespace
        fulllength = namespace+unitspace+sizespace+11
        pstr += 'Outputs\n'
        pstr += '='*fulllength
        pstr += '\n'
        pstr += 'Name'.ljust(namespace)
        pstr += 'Units'.ljust(unitspace)
        pstr += 'Size'.ljust(sizespace)
        pstr += 'Description'
        pstr += '\n'
        pstr += '-'*(namespace - whitespace)
        pstr += ' '*whitespace
        pstr += '-'*(unitspace - whitespace)
        pstr += ' '*whitespace
        pstr += '-'*(sizespace - whitespace)
        pstr += ' '*whitespace
        pstr += '-----------'
        pstr += '\n'
        for opt in self.outputs:
            pstr += opt.name.ljust(namespace)
            if opt.units is None:
                unts = 'None'
            else:
                unts = opt.units.__str__()#_repr_html_()
                # unts = unts.replace('<sup>','^')
                # unts = unts.replace('</sup>','')
                # unts = unts.replace('\[', '[')
                # unts = unts.replace('\]', ']')
            pstr += unts.ljust(unitspace)
            if opt.size is None:
                lnstr = 'None'
            else:
                lnstr = '%s'%(opt.size.__repr__())
            pstr += lnstr.ljust(sizespace)
            pstr += opt.description
            pstr += '\n'
        pstr += '\n'
        
        return pstr

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
    
    @property
    def summary(self):
        return self.getSummary()

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
    def __repr__(self):
        pstr = 'AnalysisModel( ['
        for i in range(0,len(self.outputs)):
            pstr += self.outputs[i].name
            pstr += ','
        pstr = pstr[0:-1]
        pstr += ']'
        pstr += ' == '
        pstr += 'f('
        for ipt in self.inputs:
            pstr += ipt.name
            pstr += ', '
        pstr = pstr[0:-2]
        pstr += ')'
        pstr += ' , '
        pstr = pstr[0:-2]
        pstr += '])'
        return pstr
