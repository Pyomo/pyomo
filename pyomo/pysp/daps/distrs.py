"""
  This module exports basic (and base) classese related to distributions. 
"""

import math
import os
import copy

import json
import pandas as pd
import numpy as np
import scipy.stats as sp

class Distribution:
    """ Required services common to all distributions.
    """
    def __init__(self, name = None,
                 dimension = 0):
        self.name = name
        self.dimension = dimension

    def pdf(self):
        pass
    
    def plot_pdf(self):
        pass
    def cdf(self):
        pass
    def cdf_inverse(self):
        pass

    def seed_reset(self, seed=None):
        # reset the random number seed for sampling
        pass
    
    def sample_one(self):
        # return a single sample of the proper dimension
        pass
    
    def mean(self):
        pass

    def region_expectation(self, region):
        pass
    def region_probability(self, region):
        pass

#==================================================
class ScipyDistr(Distribution):
    """ Take any scipy distr and put it in our class.
    """
    def __init__(self, scipyclassname, **kwargs):
        self.scipydistr = getattr(sp, scipyclassname)(**kwargs)
        self.dimension = 1 # generalize this... (dec 2016)

    def pdf(self, x):
        return self.scipydistr.pdf(x)

    def sample_one(self):
        return self.scipydistr.rvs()

    def seed_reset(self,seed):
        self.scipydistr.random_state = seed
        
    # etc.
     

#==================================================
class DictDistr:
    """
    This base class is intended to define something where the
    distributions live in a dictionary (perhaps of dictionaries)
    as is the standard node data definition in d2p. 
    A multivariate distr class needs to be derived that puts values
    in a dictionary of dictionaries. This is where a xi to
    param linkage lives.
    """
    def __init__(self, dict):
        self.dict = dict

    def draw_sample_into(self, fillin_dict):
        # Draw a random sample and fill in the values into the fillin_dict
        # that should have indexes matching the dict used for construction.
        pass

#==================================================
class IndepScipys(DictDistr):
    """Independent scipy distributions.
    They can be constructed with a names and args given to a constructor
    in a dictionary (of dictionaries).
    Post construction, the dictionary (reference by dict) will have ScipyDist objects
    so that samples can be drawn into a dictionarie (of dictionaries) with the same
    (jagged) indexes.
    """

    def __init__(self, dictin):
        """ The dict has the usual jagged two-level Parm index scheme, but the values
        are tuples that are (distr name, args dict). We will make a version
        of the dict where the ``values'' in the dict are the distribution objects.
        """
        self.dict = {}
        for pname in dictin:
            if isinstance(dictin[pname], dict):
                if pname not in self.dict:
                    self.dict[pname] = {}
                for pindex in dictin[pname]:
                    dname, dargs = dictin[pname][pindex]
                    self.dict[pname][pindex] = ScipyDistr(dname, **dargs)
            else:
                dname, dargs = dictin[pname]
                self.dict[pname] =  ScipyDistr(dname, **dargs)

    def draw_sample_into(self, fillin_dict):
        """Draw a random sample and put the values in the fillin_dict
        that should have indexes matching the dict used for construction.
        """
        source = self.dict  # typing aid
        dest = fillin_dict  # typing aid

        for pname in dest:
            if isinstance(dest[pname], dict):
                for pindex in dest[pname]:
                    dest[pname][pindex] = source[pname][pindex].sample_one()
            else:
                dest[pname] =  source[pname].sample_one()

#===============================================================================
# out-of-the-box utility functions
#===============================================================================
import basicclasses as bc

def dict_sampler2PySP(distrdict, \
                      TreeTemplateFileName,
                      NumScen,
                      OutDir,
                      Seed = None,
                      ScenTemplateFileName = None):
    """
    Once you have a distrdict that has the usual d2p dictionary
    indexes and values that are inputs needed to create
    scipy distribution objects (see class IndepScipys) that are
    used to draw SumScen samples (independent across params) and
    uses the TreeTemplateFilame to create a ScenarioStructure.dat
    for PySP, which is written to OutDir. A random number seed
    and the scenario template is optional.
    """
    dict_distr = IndepScipys(distrdict)

    treetemp = bc.PySP_Tree_Template()
    treetemp.read_AMPL_template(TreeTemplateFileName)
    
    if ScenTemplateFileName is not None:
        scentemp = bc.PySP_Scenario_Template()
        scentemp.read_JSON_template(ScenTemplateFileName)
    else:
        scentemp = None

    # we know we are using scipy distrubutions, so reseed accordingly
    if Seed is not None:
         np.random.seed(Seed)

    pyspscenlist = []
    rawnodelist = []  # all nodes created
    ### we should copy distrdict so it doesn't get clobbered
    localdict = copy.deepcopy(distrdict)
    for s in range(NumScen):
        # distrdict has the correct indexes, of course.
        # So now it will get the sample values, but
        # they will be overwritten next time through the loop
        # so all processing needs to be done in the loop.
        # Note: dict_distr has the distributions.
        dict_distr.draw_sample_into(localdict)
        rawnode = bc.Raw_Node_Data(filespec = None,
                                   dictin = localdict,
                                   name = "rs"+str(s+1),
                                   parentname = 'ROOT',
                                   prob = 1./float(NumScen))
        # it would not be hard to go multistage, given branching factors
        nodelistforscen = []
        nodelistforscen.append(rawnode) # single for two-stage problem
        PySPScen = bc.PySP_Scenario(bc.Raw_Scenario_Data(nodelistforscen), \
                                    scentemp)
        pyspscenlist.append(PySPScen)
        rawnodelist.append(rawnode)  # needed for the tree
        # To use an standard (AMPL format) ScenarioStructure.dat the
        # name has to be the PySPScen name because the tree constructor
        # expects that.
        fname = os.path.join(OutDir, PySPScen.name + ".json")
        with open(fname, "w") as f:
            json.dump(rawnode.valdict, f)
        
    # we have everything we need
    tree = bc.PySP_Tree(treetemp, pyspscenlist, rawnodelist)
    tree.Write_ScenarioStructure_dat_file(os.path.join(OutDir,'ScenarioStructure.dat'))
