# The experiment class is a template for making experiment lists
# to pass to parmest.  An experiment is a pyomo model "m" which has
# additional suffixes:
#   m.experiment_outputs -- which variables are experiment outputs 
#   m.unknown_parameters -- which variables are parameters to estimate
# The experiment class has only one required method:
#   get_labeled_model()
# which returns the labeled pyomo model.

class Experiment:
    def __init__(self, model=None):
        self.model = model

    def get_labeled_model(self):
        return self.model