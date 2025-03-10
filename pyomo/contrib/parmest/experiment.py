#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


class Experiment:
    """
    The experiment class is a template for making experiment lists
    to pass to parmest.

    An experiment is a Pyomo model "m" which is labeled
    with additional suffixes:
    * m.experiment_outputs which defines experiment outputs
    * m.unknown_parameters which defines parameters to estimate

    The experiment class has one required method:
    * get_labeled_model() which returns the labeled Pyomo model
    """

    def __init__(self, model=None):
        self.model = model

    def get_labeled_model(self):
        return self.model
