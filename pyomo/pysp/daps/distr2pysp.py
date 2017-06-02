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
                                 TreeTemplateFileName,\
                                 NumScen, \
                                 OutDir, \
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
        x = np.array
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
    
    distrs.dict_sampler2PySP(distrdict, \
                             TreeTemplateFileName, \
                              NumScen, \
                              OutDir, \
                              Seed)

#================================================
def json_scipy_2stage(infilename,
                      TreeTemplateFileName,\
                      NumScen, \
                      OutDir, \
                      Seed = None, \
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

    distrs.dict_sampler2PySP(distrdict, \
                             TreeTemplateFileName, \
                              NumScen, \
                              OutDir, \
                              Seed, \
                              ScenTemplateFileName)
