#  _________________________________________________________________________
#
#  Pyomo: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

__all__ = ['ModelData']

import logging
from pyomo.core.base.plugin import *

logger = logging.getLogger('pyomo.core')

class ModelData(object):
    """
    An object that manages data for a model.

    This object contains the interface routines for importing and
    exporting data from external data sources.
    """

    def __init__(self, **kwds):
        """
        Constructor
        """
        logger.warning("""DEPRECATION WARNING: ModelData objects are no longer 
supported in Pyomo, and they will be removed in Pyomo 4.0.
The DataPortal class has replaced the functionality of 
ModelData.""")

        # maps initialization data as follows: _data[namespace][symbol] -> data
        self._data={}

        # This is the data that is imported from various sources
        self._default={}

        # the model for which this data is associated. used for error
        # checking, e.g., object names must exist in the model, set 
        # dimensions must match, etc.
        self._model = kwds.pop('model', None)

        # a list of objects that implement the IDataManager interface.
        self._info=[]

        # if one desires, one can add a filename (but only one) to the 
        # ModelData object in the constructor call.
        if 'filename' in kwds:
            if not kwds['filename'] is None:
                filename = kwds['filename']
                del kwds['filename']
                self.add(filename, **kwds)
                self.read()
        elif 'data_dict' in kwds:
            if not kwds['data_dict'] is None:
                if self._model is None:
                    raise ValueError("Cannot read model data without a model")
                self._data = kwds['data_dict']
        elif len(kwds) > 0:
            raise ValueError("Unknown options: %s" % str(kwds.keys()))

    #
    # intent of this method is to add, on a component-by-component 
    # basis, initialization data to a ModelData instance. not the
    # "usual" form of data initialization.
    #
    def initialize(self, component, data, namespace=None):
        if namespace not in self._data.keys():
            self._data[namespace]={}
        self._data[namespace][component] = data

    #
    # construct data source objects associated with the input filename,
    # and store them for subsequent processing by this class. important
    # to note that this does not actually *read* the data. this is the
    # more typical use case for data initialization.
    #
    def add(self, filename, **kwds):
        # the data manager factory is keyed off the data file suffix.
        tmp = filename.split(".")[-1]
        data = DataManagerFactory(tmp) 
        if data is None:
            raise IOError("The file extension '%s' is not supported with your current Python installation.  Cannot add data from file '%s'." % (tmp, filename))
        data.initialize(filename=filename, **kwds)
        self._info.append(data)

    #
    # given a set of data sources that have been associated with this
    # ModelData object, read in and validate the data, populating the
    # the _data and _default member dictionaries.
    #
    def read(self, model=None):

        if model is not None:
            self._model=model

        if self._model is None:
            raise ValueError("Cannot read model data without a model")
        #
        # Although we want to load data in a particular order, it
        # may make sense to open/close all data resources all at once.
        # Should we do this by default, or make this a configuration option
        # for this class?
        #
        for data in self._info:
            if type(data.options.index) is tuple:
                data.options.index = list(data.options.index)
            elif not data.options.index is None and not type(data.options.index) is list:
                data.options.index = [ data.options.index ]
            if type(data.options.param) is tuple:
                data.options.param = list(data.options.param)
            elif not data.options.param is None and not type(data.options.param) is list:
                data.options.param = [ data.options.param ]

            if not data.options.index is None:
                data.options.select = data.options.index + data.options.param
                data.options.index = None
            if not data.options.set is None:
                data.options.set = getattr(self._model, data.options.set)
            if not data.options.param is None:
                if not data.options.param_name is None:
                    ans = []
                    for _param in data.options.param:
                        ans.append( getattr(self._model, data.options.param_name[_param]) )
                    data.options.param = ans
                elif type(data.options.param) in (list,tuple):
                    ans = []
                    for i in range(len(data.options.param)):
                        ans.append( getattr(self._model, data.options.param[i]) )
                    data.options.param = ans
                else:
                    data.options.param = getattr(self._model, data.options.param)
            if not data.options.index_name is None:
                data.options.index = getattr(self._model, data.options.index_name)

            if __debug__:
                logger.debug("Importing data...")

            data.open()
            if not data.read():
                print("Warning: error occured while processing %s" % str(data))
            data.close()
            if __debug__:
                logger.debug("Processing data ...")
            #
            # Update the _data and _default dictionaries with data that
            # was read in.
            #
            status = data.process(self._model, self._data, self._default)
            data.clear()
            if __debug__:
                logger.debug("Done.")

