# Get the Parm dict with distr descrs
# along with a/the template(s) and go
# all the way to PySP.
# NOTE: It seems like it might be nice to check
#  the model for the actual prescence of the Params
#  but in concrete models, use of Params is not required.
# DLW, December 2016

import os
import json
import numpy as np
import scipy.stats as sp
import distrs
import basicclasses as bc

#================================================
def indep_norms_from_data_2stage(jsonfilespec,
                                 TreeTemplateFileName,
                                 NumScen,
                                 OutDir,
                                 Seed = None):
    """ jsonfilespec gives the path to a json file that
    specifies the usual d2p node data dictionary (of dictionaries).
    The values are filenames each of which has input data that
    will be used to fit indep normals.
    Then the tree template will be used to put a ScenarioStructure.dat
    file in outdir, where the routine will also put a json 
    raw node data file for each sampled scenario (this is two stage).
    Aside: the idea is that the input data could have different
    numbers of observations for each Param and they need not be
    linked to each other across Params in any way. 
    Seed is passed through to allow random numbers to be less random.
    """

    #==========
    def d2norm(filespec):
        """ Local routine to read data and produce the norm loc and size dict.
        """
        x = np.loadtxt(filespec)
        loc, scale = sp.norm.fit(x)
        dictout = {'loc':loc, 'scale':scale}
        return 'norm', dictout

    with open(jsonfilespec) as infile:
        dictin = json.load(infile)

    # so now we make a dict suitable for the distrs.IndepScipys constructor
    distrdict = {}
    for pname in dictin:
        if isinstance(dictin[pname], dict):
            if pname not in distrdict:
                distrdict[pname] = {}
            for pindex in dictin[pname]:
                dfilespec = dictin[pname][pindex]
                distrdict[pname][pindex] = d2norm(dfilespec)
        else:
            dfilespec = dictin[pname]
            distrdict[pname] =  d2norm(dfilespec)
    
    distrs.dict_sampler2PySP(distrdict,
                             TreeTemplateFileName,
                             NumScen,
                             OutDir,
                             Seed)

def indep_norms_from_data_2stage_abstract(jsonfilespec,
                                          TreeTemplateFileName,
                                          ScenTemplateFilename,
                                          NumScen,
                                          OutDir,
                                          Seed = None):
    """
    This function should produce the required files for PySP when using an abstract
    model. This draws from a collection files which store data and fits independent
    normal distributions to each of them. It subsequently samples from them to produce
    the separate scenarios of which there are NumScen of. It then stores the scenario
    files produced in OutDir. If desired, a seed can be optionally passed in to make
    the random number generation deterministic.

    Args:
        jsonfilespec specifies a json file which contains objects which either
        map directly to files or map to objects which map to files.

        TreeTemplateFileName specifies the generic structure of the Scenario
        Tree.

        NumScen specifies the desired number of scenarios to produce

        OutDir specifies the directory to contain the output files

        Seed is an integer which can set the seed
    """

    def d2norm(filespec):
        """ Local routine to read data and produce the norm loc and size dict.
        """
        x = np.loadtxt(filespec)
        loc, scale = sp.norm.fit(x)
        dictout = {'loc':loc, 'scale':scale}
        return 'norm', dictout

    #print ("Debug: about to load infilename="+jsonfilespec)
    with open(jsonfilespec) as infile:
        dictin = json.load(infile)

    distribution_dict = {}
    for key in dictin:
        if isinstance(dictin[key], dict):
            inner_dict = dictin[key]
            distribution_dict[key] = {inner_key: d2norm(inner_dict[inner_key]) for inner_key
                                      in inner_dict}
        else:
            data_file = dictin[key]
            distribution_dict[key] = d2norm(data_file)

    distrs.dict_sampler2PySP(distribution_dict,
                             TreeTemplateFileName,
                             NumScen,
                             OutDir,
                             Seed,
                             ScenTemplateFileName=ScenTemplateFilename,
                             Abstract=True)



#================================================
def json_scipy_2stage(infilename,
                      TreeTemplateFileName,
                      NumScen,
                      OutDir,
                      Seed = None,
                      ScenTemplateFileName = None):
    """ Creates PySP inputs from independent samples.
    Reads the "standard d2p param" dict from infilename where the dictionary
    values are the (distr name, args dict) tuples to define a ScipyDistr.
    The required TreeTemplateFilename, and the optional ScenTemplateFileName,
    start_token, and end_token are "standard" d2p files. The input files for
    PySP are written to OutDir. The type for the scenario data files matches
    ScenTemplateFileName and is assumed to be json.
    NumScen dictates the number of scenarios to be sampled.
    Note: the samples are passed directly to a constructor for
    Raw_Node_Data rather than using a file.
    Another note: if you want this to be AMPL foramat, you will
    want to add start_token = None, end_token = None
    Seed is passed through to allow random numbers to be less random.
    """
    with open(infilename) as infile:
        distrdict = json.load(infile)

    distrs.dict_sampler2PySP(distrdict,
                             TreeTemplateFileName,
                              NumScen,
                              OutDir,
                              Seed,
                              ScenTemplateFileName)
