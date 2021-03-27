#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from . import pmedian

def pyomo_preprocess(**kwds):
    print( "PREPROCESSING %s"%(sorted(list(kwds.keys()))) )

def pyomo_create_model(**kwds):
    print( "CREATING MODEL %s"%(sorted(list(kwds.keys()))) )
    return pmedian.model

def pyomo_print_model(**kwds):
    print( "PRINTING MODEL %s"%(sorted(list(kwds.keys()))) )

def pyomo_print_instance(**kwds):
    print( "PRINTING INSTANCE %s"%(sorted(list(kwds.keys()))) )

def pyomo_save_instance(**kwds):
    print( "SAVE INSTANCE %s"%(sorted(list(kwds.keys()))) )

def pyomo_print_results(**kwds):
    print( "PRINTING RESULTS %s"%(sorted(list(kwds.keys()))) )

def pyomo_save_results(**kwds):
    print( "SAVING RESULTS %s"%(sorted(list(kwds.keys()))) )

def pyomo_postprocess(**kwds):
    print( "POSTPROCESSING %s"%(sorted(list(kwds.keys()))) )
