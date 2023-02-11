#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#
#  Pyomo.DoE was produced under the Department of Energy Carbon Capture Simulation 
#  Initiative (CCSI), and is copyright (c) 2022 by the software owners: 
#  TRIAD National Security, LLC., Lawrence Livermore National Security, LLC., 
#  Lawrence Berkeley National Laboratory, Pacific Northwest National Laboratory,  
#  Battelle Memorial Institute, University of Notre Dame,
#  The University of Pittsburgh, The University of Texas at Austin, 
#  University of Toledo, West Virginia University, et al. All rights reserved.
# 
#  NOTICE. This Software was developed under funding from the 
#  U.S. Department of Energy and the U.S. Government consequently retains 
#  certain rights. As such, the U.S. Government has been granted for itself
#  and others acting on its behalf a paid-up, nonexclusive, irrevocable, 
#  worldwide license in the Software to reproduce, distribute copies to the 
#  public, prepare derivative works, and perform publicly and display
#  publicly, and to permit other to do so.
#  ___________________________________________________________________________

class Measurements:
    def __init__(self, self_define_res=None, measurement_index_time=None, variance=None, ind_string='_index_'):
        """
        This class stores information on which algebraic and differential variables in the Pyomo model are considered measurements. 
        This includes the functionality to specify indices for these measurement variables. 
        For example, with a partial differential algebraic equation model, 
        these measurement index sets can specify which spatial and temporal coordinates each measurement is available. 
        Moreover, this class supports defining the covariance matrix for all measurements.

        Parameters
        ----------
        measurement_index_time:
            a ``dict``, keys are measurement variable names, 
                if there are extra index, for e.g., Var[scenario, extra_index, time]:
                    values are a dictionary, keys are its extra index, values are its measuring time points. 
                if there are no extra index, for e.g., Var[scenario, time]:
                    values are a list of measuring time point.
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
        if self_define_res:
            self.measurement_name = self_define_res
        else:
            if not measurement_index_time:
                raise AttributeError("If self-defined response names are not given, measurement time and indexes need to be defined.")

    def _model_measure_name(self):
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