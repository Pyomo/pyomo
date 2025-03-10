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

from pyomo.common.dependencies import pympler_available
from pyomo.common.collections import Bunch
import pyomo.scripting.util
from pyomo.core import ConcreteModel


def run_pyomo(options=Bunch(), parser=None):
    data = Bunch(options=options)

    if options.model.filename == '':
        parser.print_help()
        return Bunch()

    try:
        pyomo.scripting.util.setup_environment(data)

        pyomo.scripting.util.apply_preprocessing(data, parser=parser)
    except:
        # TBD: I should be able to call this function in the case of
        #      an exception to perform cleanup. However, as it stands
        #      calling finalize with its default keyword value for
        #      model(=None) results in an a different error related to
        #      task port values.  Not sure how to interpret that.
        pyomo.scripting.util.finalize(
            data, model=ConcreteModel(), instance=None, results=None
        )
        raise
    else:
        if data.error:
            # TBD: I should be able to call this function in the case of
            #      an exception to perform cleanup. However, as it stands
            #      calling finalize with its default keyword value for
            #      model(=None) results in an a different error related to
            #      task port values.  Not sure how to interpret that.
            pyomo.scripting.util.finalize(
                data, model=ConcreteModel(), instance=None, results=None
            )
            return Bunch()  # pragma:nocover

    try:
        model_data = pyomo.scripting.util.create_model(data)
    except:
        # TBD: I should be able to call this function in the case of
        #      an exception to perform cleanup. However, as it stands
        #      calling finalize with its default keyword value for
        #      model(=None) results in an a different error related to
        #      task port values.  Not sure how to interpret that.
        pyomo.scripting.util.finalize(
            data, model=ConcreteModel(), instance=None, results=None
        )
        raise
    else:
        if (
            (not options.runtime.logging == 'debug') and options.model.save_file
        ) or options.runtime.only_instance:
            pyomo.scripting.util.finalize(
                data, model=model_data.model, instance=model_data.instance, results=None
            )
            return Bunch(instance=model_data.instance)

    try:
        opt_data = pyomo.scripting.util.apply_optimizer(
            data, instance=model_data.instance
        )

        pyomo.scripting.util.process_results(
            data,
            instance=model_data.instance,
            results=opt_data.results,
            opt=opt_data.opt,
        )

        pyomo.scripting.util.apply_postprocessing(
            data, instance=model_data.instance, results=opt_data.results
        )
    except:
        # TBD: I should be able to call this function in the case of
        #      an exception to perform cleanup. However, as it stands
        #      calling finalize with its default keyword value for
        #      model(=None) results in an a different error related to
        #      task port values.  Not sure how to interpret that.
        pyomo.scripting.util.finalize(
            data, model=ConcreteModel(), instance=None, results=None
        )
        raise
    else:
        pyomo.scripting.util.finalize(
            data,
            model=model_data.model,
            instance=model_data.instance,
            results=opt_data.results,
        )

        return Bunch(
            options=options,
            instance=model_data.instance,
            results=opt_data.results,
            local=opt_data.local,
        )
