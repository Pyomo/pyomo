# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import logging
import sys
import time
from pyomo.common.download import FileDownloader, DownloadFactory
from pyomo.scripting.pyomo_parser import add_subparser


class GroupDownloader:
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
        if args.retry < 1:
            raise ValueError("--retry must be >= 1")
        results = []
        result_fmt = "[%s]  %s"

        self.downloader.cacert = args.cacert
        self.downloader.insecure = args.insecure

        logger.info(
            "As of February 9, 2023, AMPL GSL can no longer be downloaded "
            "through download-extensions. Visit https://portal.ampl.com/ "
            "to download the AMPL GSL binaries."
        )

        returncode = 0
        for target in DownloadFactory:
            attempt = 1
            while attempt <= args.retry:
                try:
                    ext = DownloadFactory(target, downloader=self.downloader)
                    if getattr(ext, "skip", bool)():
                        # If the extension has a "skip" attribute, and
                        # calling it returns Truth, then mark this as a
                        # skipped download and move on.
                        result = "SKIP"
                    else:
                        # If the extension has a __call__(), then call
                        # it.  Otherwise, assume all went well (e.g.,
                        # the registered extension was actually a
                        # function not a type -- and it was called when
                        # we "created" it in the Factory call above).
                        getattr(ext, '__call__', bool)()
                        result = " OK "

                    # Normal completion: SKIP or OK.  Either way, break
                    # out and bypass both the retry checks and any
                    # update of the returncode.
                    break

                except SystemExit:
                    _info = sys.exc_info()
                    rc = 2
                except Exception:
                    _info = sys.exc_info()
                    rc = 1

                # Note: we *only* get here if the downloader raised an exception
                _cls = 'NoneType' if _info[0] is None else _info[0].__name__
                logger.error(f"{_cls}: {_info[1]}")
                # Release the stack frame...
                _info = None

                attempt += 1
                if attempt <= args.retry:
                    logger.info(
                        f"Retrying download of '{target}' "
                        f"(attempt {attempt} of {args.retry}) "
                        f"in {args.retry_sleep} seconds"
                    )
                    time.sleep(args.retry_sleep)
                else:
                    result = "FAIL"
                    returncode |= rc

            results.append(result_fmt % (result, target))

        logger.info("Finished downloading Pyomo extensions.")
        logger.info(
            "The following extensions were downloaded:\n    " + "\n    ".join(results)
        )
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
        add_help=True,
        description='This downloads all registered (compiled) extension modules',
    )
)

_parser.add_argument(
    '-r',
    '--retry',
    action='store',
    type=int,
    dest='retry',
    default=1,
    help="Total number of attempts for each download (must be >= 1)",
)

_parser.add_argument(
    '-s',
    '--retry-sleep',
    action='store',
    type=int,
    dest='retry_sleep',
    default=15,
    help="Seconds to sleep before retrying the download",
)
