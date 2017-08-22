"""
  This module exports basic (and base) classese related to distributions. 
"""

import math
import os
import copy

import json
import pandas as pd
import numpy as np
import scipy.integrate
import scipy.optimize
import scipy.stats as sp

import cutpoint_set

class Distribution:
    """ Required services common to all distributions.
    """

    def pdf(self, x):
        pass
    
    def plot_pdf(self):
        pass

    def cdf(self, x):
        return scipy.integrate.quad(self.pdf, -np.inf, x) # For univariate only

    def cdf_inverse(self, y):
        return scipy.optimize.newton(lambda x: (y - self.cdf(x)) ** 2, 0) # Bad implementation, just an idea

    def seed_reset(self, seed=None):
        # reset the random number seed for sampling
        pass
    
    def sample_one(self):
        # return a single sample of the proper dimension
        pass
    
    def mean(self):
        pass

    def region_expectation(self, a, b):
        return scipy.integrate.quad(lambda x: x * self.pdf(x), a, b)

    def region_probability(self, a, b):
        return scipy.integrate.quad(self.pdf, a, b)

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

    def cdf(self, x):
        return self.scipydistr.cdf(x)

    def cdf_inverse(self, y):
        return self.scipydistr.ppf(y)

    def region_expectation(self, a, b):
        return self.scipydistr.expect(lb=a, ub=b)

    def region_probability(self, a, b):
        return self.scipydistr.expect(func=lambda x: 1, lb=a, ub=b)

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

    def pick_representative_points(self, fillin_dict, rectangle_dict):
        source = self.dict  # typing aid
        dest = fillin_dict  # typing aid

        for pname in dest:
            if isinstance(dest[pname], dict):
                for pindex in dest[pname]:
                    a, b = rectangle_dict[pname][pindex]
                    dest[pname][pindex] = source[pname][pindex].region_expectation(a, b)
            else:
                a, b = rectangle_dict[pname]
                dest[pname] =  source[pname].region_expectation(a, b)

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

def dict_sampler2PySP(distrdict,
                      TreeTemplateFileName,
                      NumScen,
                      OutDir,
                      Seed = None,
                      ScenTemplateFileName = None,
                      Abstract = False):
    """
    Once you have a distrdict that has the usual d2p dictionary
    indexes and values that are inputs needed to create
    scipy distribution objects (see class IndepScipys) that are
    used to draw NumScen samples (independent across params) and
    uses the TreeTemplateFilename to create a ScenarioStructure.dat
    for PySP, which is written to OutDir. A random number seed
    and the scenario template is optional.
    """
    dict_distr = IndepScipys(distrdict)

    treetemp = bc.PySP_Tree_Template()
    treetemp.read_AMPL_template(TreeTemplateFileName)
    
    if ScenTemplateFileName is not None:
        scentemp = bc.PySP_Scenario_Template()
        scentemp.read_AMPL_template_with_tokens(ScenTemplateFileName)
    else:
        scentemp = None

    # we know we are using scipy distributions, so reseed accordingly
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
        PySPScen = bc.PySP_Scenario(bc.Raw_Scenario_Data(nodelistforscen),
                                    scentemp)
        pyspscenlist.append(PySPScen)
        rawnodelist.append(rawnode)  # needed for the tree
        # To use an standard (AMPL format) ScenarioStructure.dat the
        # name has to be the PySPScen name because the tree constructor
        # expects that.
        if not Abstract:
            fname = os.path.join(OutDir, PySPScen.name + ".json")
            with open(fname, "w") as f:
                json.dump(rawnode.valdict, f)
        else:
            fname = os.path.join(OutDir, PySPScen.name + ".dat")
            with open(fname, "wt") as f:
                for line in PySPScen.scenariodata:
                    f.write(line)
        
    # we have everything we need
    tree = bc.PySP_Tree(treetemp, pyspscenlist, rawnodelist)
    tree.Write_ScenarioStructure_dat_file(os.path.join(OutDir,'ScenarioStructure.dat'))


def dict_representative_points_scenarios(distrdict,
                      TreeTemplateFileName,
                      CutPointFileName,
                      NumScen,
                      OutDir,
                      Seed=None,
                      ScenTemplateFileName=None,
                      Abstract=False):
    """
    Once you have a distrdict that has the usual d2p dictionary
    indexes and values that are inputs needed to create
    scipy distribution objects (see class IndepScipys) that are
    used to draw NumScen samples (independent across params) and
    uses the TreeTemplateFilename to create a ScenarioStructure.dat
    for PySP, which is written to OutDir. A random number seed
    and the scenario template is optional.
    """
    dict_distr = IndepScipys(distrdict)

    treetemp = bc.PySP_Tree_Template()
    treetemp.read_AMPL_template(TreeTemplateFileName)

    cutpoint_sets = cutpoint_set.parse_cutpoint_file(CutPointFileName)

    if ScenTemplateFileName is not None:
        scentemp = bc.PySP_Scenario_Template()
        scentemp.read_AMPL_template_with_tokens(ScenTemplateFileName)
    else:
        scentemp = None

    # we know we are using scipy distributions, so reseed accordingly
    if Seed is not None:
        np.random.seed(Seed)

    pyspscenlist = []
    rawnodelist = []  # all nodes created
    ### we should copy distrdict so it doesn't get clobbered
    localdict = copy.deepcopy(distrdict)
    rectangledict = copy.deepcopy(distrdict)

    interval_count = sum(len(cpt_set.intervals) for cpt_set in cutpoint_sets)

    for cpt_set in cutpoint_sets:
        for name in cpt_set.cutpoint_names:
            # distrdict has the correct indexes, of course.
            # So now it will get the sample values, but
            # they will be overwritten next time through the loop
            # so all processing needs to be done in the loop.
            # Note: dict_distr has the distributions.
            interval = cpt_set.get(name)
            for pname in rectangledict:
                if isinstance(rectangledict[pname], dict):
                    for pindex in rectangledict[pname]:
                        rectangledict[pname][pindex] = interval
                else:
                    rectangledict[pname] = interval

            dict_distr.pick_representative_points(localdict, rectangledict)
            rawnode = bc.Raw_Node_Data(filespec=None,
                                       dictin=localdict,
                                       name=name,
                                       parentname='ROOT',
                                       prob=1. / interval_count)
            # it would not be hard to go multistage, given branching factors
            nodelistforscen = []
            nodelistforscen.append(rawnode)  # single for two-stage problem
            PySPScen = bc.PySP_Scenario(bc.Raw_Scenario_Data(nodelistforscen),
                                        scentemp)
            pyspscenlist.append(PySPScen)
            rawnodelist.append(rawnode)  # needed for the tree
            # To use an standard (AMPL format) ScenarioStructure.dat the
            # name has to be the PySPScen name because the tree constructor
            # expects that.
            if not Abstract:
                fname = os.path.join(OutDir, PySPScen.name + ".json")
                with open(fname, "w") as f:
                    json.dump(rawnode.valdict, f)
            else:
                fname = os.path.join(OutDir, PySPScen.name + ".dat")
                with open(fname, "wt") as f:
                    for line in PySPScen.scenariodata:
                        f.write(line)

    # we have everything we need
    tree = bc.PySP_Tree(treetemp, pyspscenlist, rawnodelist)
    tree.Write_ScenarioStructure_dat_file(os.path.join(OutDir, 'ScenarioStructure.dat'))