import pmedian

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
