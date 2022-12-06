from pyomo.common.dependencies import (
    numpy as np, numpy_available,
    pandas as pd, pandas_available,
    scipy, scipy_available,
    matplotlib as plt, matplotlib_available,
)


class Measurements:
    def __init__(self, measurement_index_time, variance=None, ind_string='_index_'):
        """This class stores information on which algebraic and differential variables in the Pyomo model are considered measurements. 
        This includes the functionality to specify indices for these measurement variables. 
        For example, with a partial differential algebraic equation model, 
        these measurement index sets can specify which spatial and temporal coordinates each measurement is available. 
        Moreover, this class supports defining the covariance matrix for all measurements.

        Parameters
        ----------
        measurement_index_time:
            a ``dict``, keys are measurement variable names, values are a dictionary, keys are its extra index,
            values are its measuring time points, values are a list of measuring time point if there is no extra index for this measurement.
            For e.g., for the kinetics illustrative example, it should be {'C':{'CA':[0,1,..], 'CB':[0,2,...]}, 'k':[0,4,..]},
            so the measurements are C[scenario, 'CA', 0]..., k[scenario, 0]....
        variance:
            a ``dict``, keys are measurement variable names, values are a dictionary, keys are its extra index,
            values are its variance (a scalar number), values are its variance if there is no extra index for this measurement.
            For e.g., for the kinetics illustrative example, it should be {'C':{'CA': 10, 'CB': 1, 'CC': 2}}.
            If given None, the default is {'C':{'CA': 1, 'CB': 1, 'CC': 1}}.
        ind_string:
            a ''string'', used to flatten the name of variables and extra index. Default is '_index_'.
            For e.g., for {'C':{'CA': 10, 'CB': 1, 'CC': 2}}, the reformulated name is 'C_index_CA'.
        """
        self.measurement_all_info = measurement_index_time
        self.ind_string = ind_string
        # a list of measurement names
        self.measurement_name = list(measurement_index_time.keys())
        # begin flatten
        self.__name_and_index_generator(self.measurement_all_info)
        self.__generate_flatten_name(self.name_and_index)
        self.__generate_variance(self.flatten_measure_name, variance, self.name_and_index)
        self.__generate_flatten_timeset(self.measurement_all_info, self.flatten_measure_name, self.name_and_index)
        self.__model_measure_name()

        # generate the overall measurement time points set, including the measurement time for all measurements
        flatten_timepoint = list(self.flatten_measure_timeset.values())
        overall_time = []
        for i in flatten_timepoint:
            overall_time += i
            timepoint_overall_set = list(set(overall_time))
        self.timepoint_overall_set = timepoint_overall_set


    def __name_and_index_generator(self, all_info):
        """
        Generate a dictionary, keys are the variable names, values are the indexes of this variable.
        For e.g., name_and_index = {'C': ['CA', 'CB', 'CC']}
        Parameters
        ---------
        all_info: a dictionary, keys are measurement variable names,
                values are a dictionary, keys are its extra index, values are its measuring time points
                values are a list of measuring time point if there is no extra index for this measurement
            Note: all_info can be the self.measurement_all_info, but does not have to be it.
        """
        measurement_name = list(all_info.keys())
        # a list of measurement extra indexes
        measurement_extra_index = []
        # check if the measurement has extra indexes
        for i in measurement_name:
            if type(all_info[i]) is dict:
                index_list = list(all_info[i].keys())
                measurement_extra_index.append(index_list)
            elif type(all_info[i]) is list:
                measurement_extra_index.append(None)
        # a dictionary, keys are measurement names, values are a list of extra indexes
        self.name_and_index = dict(zip(measurement_name, measurement_extra_index))

    def __generate_flatten_name(self, measure_name_and_index):
        """Generate measurement flattened names
        Parameters
        ----------
        measure_name_and_index: a dictionary, keys are measurement names, values are lists of extra indexes

        Returns
        ------
        jac_involved_name: a list of flattened measurement names
        """
        flatten_names = []
        for j in measure_name_and_index.keys():
            if measure_name_and_index[j] is not None: # if it has extra index
                for ind in measure_name_and_index[j]:
                    flatten_name = j + self.ind_string + str(ind)
                    flatten_names.append(flatten_name)
            else:
                flatten_names.append(j)

        self.flatten_measure_name = flatten_names

    def __generate_variance(self, flatten_measure_name, variance, name_and_index):
        """Generate the variance dictionary
        Parameters
        ----------
        flatten_measure_name: flattened measurement names. For e.g., flattenning {'C':{'CA': 10, 'CB': 1, 'CC': 2}} will be 'C_index_CA', ..., 'C_index_CC'.
        variance:
            a ``dict``, keys are measurement variable names, values are a dictionary, keys are its extra index name,
            values are its variance as a scalar number.
            For e.g., for the kinetics illustrative example, it should be {'C':{'CA': 10, 'CB': 1, 'CC': 2}}.
            If given None, the default is {'C':{'CA': 1, 'CB': 1, 'CC': 1}}.
            If there is no extra index, it is a dict, keys are measurement variable names, values are its variance as a scalar number.
        name_and_index:
            a dictionary, keys are measurement names, values are a list of extra indexes.
        """
        flatten_variance = {}
        for i in flatten_measure_name:
            if variance is None:
                flatten_variance[i] = 1
            else:
                # split the flattened name if needed
                if self.ind_string in i:
                    measure_name = i.split(self.ind_string)[0]
                    measure_index = i.split(self.ind_string)[1]
                    if type(name_and_index[measure_name][0]) is int:
                        measure_index = int(measure_index)
                    flatten_variance[i] = variance[measure_name][measure_index]
                else:
                    flatten_variance[i] = variance[i]
        self.flatten_variance = flatten_variance

    def __generate_flatten_timeset(self, all_info, flatten_measure_name,name_and_index):
        """
        Generate flatten variables timeset. Return a dict where keys are the flattened variable names,
        values are a list of measurement time.

        """
        flatten_measure_timeset = {}
        for i in flatten_measure_name:
            # split the flattened name if needed
            if self.ind_string in i:
                measure_name = i.split(self.ind_string)[0]
                measure_index = i.split(self.ind_string)[1]
                if type(name_and_index[measure_name][0]) is int:
                    measure_index = int(measure_index)
                flatten_measure_timeset[i] = all_info[measure_name][measure_index]
            else:
                flatten_measure_timeset[i] = all_info[i]
        self.flatten_measure_timeset = flatten_measure_timeset

    def __model_measure_name(self):
        """Return pyomo string name
        """
        # store pyomo string name
        measurement_names = []
        # loop over measurement name
        for mname in self.flatten_measure_name:
            # check if there is extra index
            if self.ind_string in mname:
                measure_name = mname.split(self.ind_string)[0]
                measure_index = mname.split(self.ind_string)[1]
                for tim in self.flatten_measure_timeset[mname]:
                    # get the measurement name in the model
                    measurement_name = measure_name + '[0,' + measure_index + ',' + str(tim) + ']'
                    measurement_names.append(measurement_name)
            else:
                for tim in self.flatten_measure_timeset[mname]:
                    # get the measurement name in the model
                    measurement_name = mname + '[0,' + str(tim) + ']'
                    measurement_names.append(measurement_name)
        self.model_measure_name = measurement_names

    def SP_measure_name(self, j, t,scenario_all=None, p=None, mode='sequential_finite', legal_t=True):
        """Return pyomo string name for different modes
        Arguments
        ---------
        j: flatten measurement name
        t: time 
        scenario_all: all scenario object, only needed for simultaneous finite mode
        p: parameter, only needed for simultaneous finite mode
        mode: mode name, can be 'simultaneous_finite' or 'sequential_finite'
        legal_t: if the time point is legal for this measurement. default is True
        
        Return
        ------
        up_C, lo_C: two measurement pyomo string names for simultaneous mode
        legal_t: if the time point is legal for this measurement 
        string_name: one measurement pyomo string name for sequential 
        """
        if mode=='simultaneous_finite':
            # check extra index
            if self.ind_string in j:
                measure_name = j.split(self.ind_string)[0]
                measure_index = j.split(self.ind_string)[1]
                if type(self.name_and_index[measure_name][0]) is str:
                    measure_index = '"' + measure_index + '"'
                if t in self.flatten_measure_timeset[j]:
                    up_C = 'm.' + measure_name + '[' + str(scenario_all['jac-index'][p][0]) + ',' + measure_index + ',' + str(t) + ']'
                    lo_C = 'm.' + measure_name + '[' + str(scenario_all['jac-index'][p][1]) + ',' + measure_index + ',' + str(t) + ']'
                else:
                    legal_t = False
            else:
                up_C = 'm.' + j + '[' + str(scenario_all['jac-index'][p][0]) + ',' + str(t) + ']'
                lo_C = 'm.' + j + '[' + str(scenario_all['jac-index'][p][1]) + ',' + str(t) + ']'

            return up_C, lo_C, legal_t
        
        elif mode == 'sequential_finite':
            if self.ind_string in j:
                measure_name = j.split(self.ind_string)[0]
                measure_index = j.split(self.ind_string)[1]
                if type(self.name_and_index[measure_name][0]) is str:
                    measure_index = '"' + measure_index + '"'
                if t in self.flatten_measure_timeset[j]:
                    string_name = 'mod.' + measure_name + '[0,' + str((measure_index)) + ',' + str(t) + ']'
            else:
                string_name = 'mod.' + j + '[0,' + str(t) + ']'

            return string_name


    def check_subset(self,subset, throw_error=True, valid_subset=True):
        """
        Check if the subset is correctly defined with right name, index and time.

        subset:
            a ''dict'' where measurement name and index are involved in jacobian calculation
        throw_error:
            if the given subset is not a subset of the measurement set, throw error message
        """
        flatten_subset = subset.flatten_measure_name
        flatten_timeset = subset.flatten_measure_timeset
        # loop over subset measurement names
        for i in flatten_subset:
            # check if subset measurement names are in the overall measurement names
            if i not in self.flatten_measure_name:
                valid_subset = False
                if throw_error:
                    raise ValueError('This is not a legal subset of the measurement overall set!')
            else:
                # check if subset measurement timepoints are in the overall measurement timepoints
                for t in flatten_timeset[i]:
                    if t not in self.flatten_measure_timeset[i]:
                        valid_subset = False
                        if throw_error:
                            raise ValueError('The time of {} is not included as measurements before.'.format(t))
        return valid_subset