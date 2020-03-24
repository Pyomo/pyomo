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
import os
import platform
import sys
from pyomo.common import Library
from pyomo.common.download import FileDownloader

logger = logging.getLogger('pyomo.common')

# These URLs were retrieved from
#     https://ampl.com/resources/extended-function-library/
urlmap = {
    'linux':   'https://ampl.com/NEW/amplgsl/amplgsl.linux-intel%s.zip',
    'windows': 'https://ampl.com/NEW/amplgsl/amplgsl.mswin%s.zip',
    'cygwin':  'https://ampl.com/NEW/amplgsl/amplgsl.mswin%s.zip',
    'darwin':  'https://ampl.com/NEW/amplgsl/amplgsl.macosx%s.zip'
}

def find_GSL():
    # FIXME: the GSL interface is currently broken in PyPy:
    if platform.python_implementation().lower().startswith('pypy'):
        return None
    return Library('amplgsl.dll').path()

def get_gsl(downloader):
    system, bits = downloader.get_sysinfo()
    url = downloader.get_platform_url(urlmap) % (bits,)

    downloader.set_destination_filename(os.path.join('lib', 'amplgsl.dll'))

    logger.info("Fetching GSL from %s and installing it to %s"
                % (url, downloader.destination()))

    downloader.get_binary_file_from_zip_archive(url, 'amplgsl.dll')

def main(argv):
    downloader = FileDownloader()
    downloader.parse_args(argv)
    get_gsl(downloader)

if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    try:
        main(sys.argv[1:])
    except Exception as e:
        print(e.message or str(e))
        print("Usage: %s [--insecure] [target]" % os.path.basename(sys.argv[0]))
        sys.exit(1)
