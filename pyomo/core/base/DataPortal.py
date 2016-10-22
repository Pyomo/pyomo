#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ['DataPortal']

import logging
from pyomo.core.base.plugin import *

logger = logging.getLogger('pyomo.core')


class DataPortal(object):
    """
    An object that manages loading and storing data from
    external data sources.  This object interfaces to plugins that
    manipulate the data in a manner that is dependent on the
    data format.

    Note that data is organized as follows:
        data[namespace][symbol][index] -> value
    The default namespace is None.

    Constructor Arguments:
        model       The model for which this data is associated.  This is
                        used for error checing (e.g. object names must
                        exist in the model, set dimensions must match, etc.)
        filename    A file that is loaded
        data_dict   A dictionarity used to initialize the data map
                        in this object.
    """

    def __init__(self, *args, **kwds):
        """
        Constructor
        """
        if len(args) > 0:
            raise RuntimeError("Unexpected constructor argument for a DataPortal object")

        # Initialize this object with no data manager
        self._data_manager = None

        # Map initialization data as follows: _data[namespace][symbol] -> data
        self._data={}

        # This is the data that is imported from various sources
        self._default={}

        # Get the model for which this data is associated.
        self._model = kwds.pop('model', None)

        # Load data from a file ...
        if 'filename' in kwds:
            filename = kwds.pop('filename')
            self.connect(filename=filename, **kwds)
            self.load()
            self.disconnect()
        # Or load data from a dictionary
        elif 'data_dict' in kwds:
            data = kwds.pop('data_dict')
            if not data is None:
                self._data = data
        elif len(kwds) > 0:
            raise ValueError("Unknown options: %s" % str(kwds.keys()))

    def connect(self, **kwds):
        """
        Construct a data manager object that is associated with the input source.
        This data manager is used to process future data imports and exports.

        Arguments:
            filename    A file that is loaded
            server      The name of the remote server that hosts the data

        Other keyword arguments are passed to the data manager object.
        """
        if not self._data_manager is None:
            self._data_manager.close()
        data = kwds.get('using',None)
        if data is None:
            data = kwds.get('filename',None)
        if data is None:
            data = kwds.get('server',None)
        if '.' in data:
            tmp = data.split(".")[-1]
        else:
            tmp = data
        self._data_manager = DataManagerFactory(tmp)
        if type(self._data_manager) is UnknownDataManager:
            raise IOError("Unknown file format '%s'" % tmp)
        self._data_manager.initialize(**kwds)
        self._data_manager.open()

    def disconnect(self):
        """
        Close the data manager object that is associated with the input source.
        """
        self._data_manager.close()
        self._data_manager = None

    def load(self, **kwds):
        """
        Import data from an external data source.

        Arguments:
            model       The model for which this data is associated.
            set         TODO
            param       TODO
            index       TODO

            format      TODO
            select      TODO
            filename    TODO
            namespace   TODO

        Other keyword arguments are passed to connect().
        """
        if __debug__ and logger.isEnabledFor(logging.DEBUG):        #pragma:nocover
            logger.debug("Loading data...")
        #
        # Process arguments
        #
        _model = kwds.pop('model', None)
        if not _model is None:
            self._model=_model
        # If _disconnect is True, then disconnect the data manager after we load data
        _disconnect=False
        if self._data_manager is None:
            #
            # Start a new connection
            #
            self.connect(**kwds)
            _disconnect=True
        elif len(kwds) > 0:
            #
            # We are continuing to store using an existing connection.
            #
            # Q: Should we reinitialize?  The semantic difference between
            # initialize() and add_options() aren't clear.
            self._data_manager.add_options(**kwds)
        #
        # Preprocess the command-line options
        #
        self._preprocess_options()
        #
        # Read from data manager into self._data and self._default
        #
        if __debug__ and logger.isEnabledFor(logging.DEBUG):        #pragma:nocover
            logger.debug("Processing data ...")
        self._data_manager.read()
        status = self._data_manager.process(self._model, self._data, self._default)
        self._data_manager.clear()
        #
        # Disconnect
        #
        if _disconnect:
            self.disconnect()
        if __debug__ and logger.isEnabledFor(logging.DEBUG):        #pragma:nocover
            logger.debug("Done.")

    def store(self, **kwds):
        """
        Export data from to an external data source.

        Arguments:
            model       The model for which this data is associated.
            set         TODO
            param       TODO
            index       TODO
            data        TODO

            columns     TODO
            filename    TODO

        Other keyword arguments are passed to connect().
        """
        if __debug__ and logger.isEnabledFor(logging.DEBUG):        #pragma:nocover
            logger.debug("Storing data...")
        #
        # Process arguments
        #
        _model = kwds.pop('model', None)
        if not _model is None:
            self._model=_model
        # If _disconnect is True, then disconnect the data manager after we load data
        _disconnect=False
        if self._data_manager is None:
            self.connect(**kwds)
            _disconnect=True
        elif len(kwds) > 0:
            # Q: Should we reinitialize?  The semantic difference between
            # initialize() and add_options() aren't clear.
            self._data_manager.add_options(**kwds)
        #
        # Preprocess the command-line options
        #
        self._preprocess_options()
        self._load_data_from_model()
        #
        # Write from self._data
        #
        self._data_manager.write(self._data)
        #
        # Disconnect
        #
        if _disconnect:
            self.disconnect()
        if __debug__ and logger.isEnabledFor(logging.DEBUG):        #pragma:nocover
            logger.debug("Done.")

    def data(self, name=None, namespace=None):
        """
        Return the data associated with a symbol and namespace

        If the data is simply a value, then the value is returned.  Otherwise,
        a dictionary is returned that maps index to value.
        """
        if not namespace in self._data:
            raise IOError("Unknown namespace '%s'" % str(namespace))
        if name is None:
            return self._data[namespace]
        ans = self._data[namespace][name]
        if None in ans:
            # The data is a simple value
            return ans[None]
        return ans

    def __getitem__(self, *args):
        """
        Return the specified data value:
            dp = DataPortal()
            dp[name]
            dp[namespace, name]
        """
        if type(args[0]) is tuple or type(args[0]) is list:
            assert(len(args) == 1)
            args = args[0]
        if len(args) > 2:
            raise IOError("Must specify data name:  DataPortal[name] or Data[namespace, name]")
        elif len(args) == 2:
            namespace = args[0]
            name = args[1]
        else:
            namespace=None
            name = args[0]

        ans = self._data[namespace][name]
        if None in ans:
            # The data is a simple value
            return ans[None]
        return ans

    def __setitem__(self, index, value):
        if not None in self._data:
            self._data[None] = {}
        self._data[None][index] = value

    def namespaces(self):
        """
        Return an iterator for the data namespaces
        """
        for key in self._data:
            yield key

    def keys(self, namespace=None):
        """
        Return an iterator of the data keys in
        the specified namespace
        """
        for key in self._data[namespace]:
            yield key

    def values(self, namespace=None):
        """
        Return an iterator of the data values in
        the specified namespace
        """
        for key in self._data[namespace]:
            ans = self._data[namespace][key]
            if None in ans:
                yield ans[None]
            else:
                yield ans

    def items(self, namespace=None):
        """
        Return an iterator of (name, value) tuples from the data in
        the specified namespace
        """
        for key in self._data[namespace]:
            ans = self._data[namespace][key]
            if None in ans:
                yield key, ans[None]
            else:
                yield key, ans

    def _preprocess_options(self):
        """
        Preprocess the options for a data manager.
        """
        options = self._data_manager.options
        #
        if options.data is None and (not options.set is None or not options.param is None or not options.index is None):
            #
            # Set options.data to a list of elements of the options.set,
            # options.param and options.index values.
            #
            options.data = []
            if not options.set is None:
                assert(type(options.set) not in (list, tuple))
                options.data.append(options.set)
                #
                # The set option should not be a list or tuple.
                #
                #if type(options.set) in (list,tuple):
                #    for item in options.set:
                #        options.data.append(item)
                #else:
                #    options.data.append(options.set)
            if not options.index is None:
                options.data.append(options.index)
            if not options.param is None:
                if type(options.param) in (list,tuple):
                    for item in options.param:
                        options.data.append(item)
                else:
                    options.data.append(options.param)
        #
        if options.data is None:
            return
        #
        if type(options.data) in (list, tuple):
            #
            # If options.data is a list/tuple, then
            # process it to get the names of the
            # elements.  Thus, if a component is included
            # in options.data, then it is replaced by its name.
            #
            ans = []
            for item in options.data:
                try:
                    ans.append(item.local_name)
                    self._model = item.model()
                except:
                    ans.append(item)
            options.data = ans
        else:
            #
            # If options.data is a single value, then we assume that
            # it is a component.  Reset its value to the value of
            # the component name.
            #
            try:
                self._model = options.data.model()
                options.data = [ self._data_manager.options.data.local_name ]
            except:
                pass

    def _load_data_from_model(self):
        """
        Load model data into self._data
        """
        if self._data_manager.options.data is None or self._model is None:
            return
        for name in self._data_manager.options.data:
            c = getattr(self._model, name)
            try:
                self._data[name] = c.data()
            except:
                self._data[name] = c.extract_values()

