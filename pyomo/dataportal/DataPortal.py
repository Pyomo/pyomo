#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['DataPortal']

import logging
from pyomo.common.log import is_debug_set
from pyomo.dataportal.factory import DataManagerFactory, UnknownDataManager

logger = logging.getLogger('pyomo.core')


class DataPortal(object):
    """
    An object that manages loading and storing data from external
    data sources.  This object interfaces to plugins that manipulate
    the data in a manner that is dependent on the data format.

    Internally, the data in a DataPortal object is organized as follows::

        data[namespace][symbol][index] -> value

    All data is associated with a symbol name, which may be indexed,
    and which may belong to a namespace.  The default namespace is
    :const:`None`.

    Args:
        model: The model for which this data is associated.  This is
            used for error checking (e.g. object names must
            exist in the model, set dimensions must match, etc.).
            Default is :const:`None`.
        filename (str): A file from which data is loaded.  Default
            is :const:`None`.
        data_dict (dict): A dictionary used to initialize the data 
            in this object.  Default is :const:`None`.
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

        Args:
            filename (str): A filename that specifies the data source.  
                Default is :const:`None`.
            server (str): The name of the remote server that hosts the data.  
                Default is :const:`None`.
            using (str): The name of the resource used to load the data.
                Default is :const:`None`.

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
        Close the data manager object that is associated with the
        input source.
        """
        self._data_manager.close()
        self._data_manager = None

    def load(self, **kwds):
        """
        Import data from an external data source.

        Args:
            model: The model object for which this data is associated.
                Default is :const:`None`.

        Other keyword arguments are passed to the :func:`connect()` method.
        """
        if is_debug_set(logger):        #pragma:nocover
            logger.debug("Loading data...")
        #
        # Process arguments
        #
        _model = kwds.pop('model', None)
        if not _model is None:
            self._model=_model
        #
        # If _disconnect is True, then disconnect the data 
        # manager after we load data
        #
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
            #
            self._data_manager.add_options(**kwds)
        #
        # Preprocess the command-line options
        #
        self._preprocess_options()
        #
        # Read from data manager into self._data and self._default
        #
        if is_debug_set(logger):        #pragma:nocover
            logger.debug("Processing data ...")
        self._data_manager.read()
        status = self._data_manager.process(self._model, self._data, self._default)
        self._data_manager.clear()
        #
        # Disconnect
        #
        if _disconnect:
            self.disconnect()
        if is_debug_set(logger):        #pragma:nocover
            logger.debug("Done.")

    def store(self, **kwds):
        """
        Export data to an external data source.

        Args:
            model: The model object for which this data is associated.
                Default is :const:`None`.

        Other keyword arguments are passed to the :func:`connect()` method.
        """
        if is_debug_set(logger):        #pragma:nocover
            logger.debug("Storing data...")
        #
        # Process arguments
        #
        _model = kwds.pop('model', None)
        if not _model is None:
            self._model=_model
        #
	    # If _disconnect is True, then disconnect the data manager
	    # after we load data
        #
        _disconnect=False
        if self._data_manager is None:
            self.connect(**kwds)
            _disconnect=True
        elif len(kwds) > 0:
            #
            # Q: Should we reinitialize?  The semantic difference between
            # initialize() and add_options() aren't clear.
            #
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
        if is_debug_set(logger):        #pragma:nocover
            logger.debug("Done.")

    def data(self, name=None, namespace=None):
        """
	    Return the data associated with a symbol and namespace

        Args:
            name (str): The name of the symbol that is returned.
                Default is :const:`None`, which indicates that the
                entire data in the namespace is returned.
            namespace (str): The name of the namespace that is accessed.
                Default is :const:`None`.

        Returns:
            If ``name`` is :const:`None`, then the dictionary for
            the namespace is returned.  Otherwise, the data 
            associated with ``name`` in given namespace is returned.
            The return value is a constant if :const:`None` if 
            there is a single value in the symbol dictionary, and otherwise
            the symbol dictionary is returned.            
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
        Return the specified data value.  

        If a single argument is given, then this is the symbol name::

            dp = DataPortal()
            dp[name]

        If a two arguments are given, then the first is the namespace and
        the second is the symbol name::

            dp = DataPortal()
            dp[namespace, name]

        Args:
            *args (str): A tuple of arguents.

        Returns:
            If a single argument is given, then the data associated
            with that symbol in the namespace :const:`None` is returned.
            If two arguments are given, then the data associated with
            symbol in the given namespace is returned.
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

    def __setitem__(self, name, value):
        """
        Set the value of ``name`` with the given value.

        Args:
            name (str): The name of the symbol that is set.
            value: The value of the symbol.
        """
        if not None in self._data:
            self._data[None] = {}
        self._data[None][name] = value

    def namespaces(self):
        """
        Return an iterator for the namespaces in the data portal.

        Yields:
            A string name for the next namespace.
        """
        for key in self._data:
            yield key

    def keys(self, namespace=None):
        """
        Return an iterator of the data keys in
        the specified namespace.

        Yields:
            A string name for the next symbol in the specified namespace.
        """
        for key in self._data[namespace]:
            yield key

    def values(self, namespace=None):
        """
        Return an iterator of the data values in
        the specified namespace.

        Yields:
            The data value for the next symbol in the specified namespace.
            This may be a simple value, or a dictionary of values.
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
        the specified namespace.

        Yields:
            The next (name, value) tuple in the namespace.  If the symbol
            has a simple data value, then that is included in the tuple.
            Otherwise, the tuple includes a dictionary mapping
            symbol indices to values.
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

