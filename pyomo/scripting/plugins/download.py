#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
import sys
from pyomo.common.download import FileDownloader, DownloadFactory
from pyomo.scripting.pyomo_parser import add_subparser

class GroupDownloader(object):
    def __init__(self):
        self.downloader = FileDownloader()

    def create_parser(self, parser):
        return self.downloader.create_parser(parser)

    def call(self, args, unparsed):
        logger = logging.getLogger('pyomo.common')
        original_level = logger.level
        logger.setLevel(logging.INFO)
        try:
            return self._call_impl(args, unparsed, logger)
        finally:
            logger.setLevel(original_level)

    def _call_impl(self, args, unparsed, logger):
        results = []
        result_fmt = "[%s]  %s"
        returncode = 0
        self.downloader.cacert = args.cacert
        self.downloader.insecure = args.insecure
        for target in DownloadFactory:
            try:
                ext = DownloadFactory(target, downloader=self.downloader)
                if hasattr(ext, 'skip') and ext.skip():
                    result = 'SKIP'
                elif hasattr(ext, '__call__'):
                    ext()
                    result = ' OK '
                else:
                    # Extension was a simple function and already ran
                    result = ' OK '
            except SystemExit:
                _info = sys.exc_info()
                _cls = str(_info[0].__name__ if _info[0] is not None
                           else "NoneType") + ": "
                logger.error(_cls + str(_info[1]))
                result = 'FAIL'
                returncode |= 2
            except:
                _info = sys.exc_info()
                _cls = str(_info[0].__name__ if _info[0] is not None
                           else "NoneType") + ": "
                logger.error(_cls + str(_info[1]))
                result = 'FAIL'
                returncode |= 1
            results.append(result_fmt % (result, target))
        logger.info("Finished downloading Pyomo extensions.")
        logger.info(
            "The following extensions were downloaded:\n    " +
            "\n    ".join(results))
        return returncode


#
# Add a subparser for the download-extensions command
#
_group_downloader = GroupDownloader()
_parser = _group_downloader.create_parser(
    add_subparser(
        'download-extensions',
        func=_group_downloader.call,
        help='Download compiled extension modules',
        add_help=False,
        description='This downloads all registered (compiled) extension modules'
    ))

