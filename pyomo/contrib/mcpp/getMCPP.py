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
import sys
from pyomo.common.download import FileDownloader

logger = logging.getLogger('pyomo.common')


def get_mcpp(downloader):
    url = 'https://github.com/omega-icl/mcpp/archive/master.zip'

    downloader.set_destination_filename(os.path.join('src', 'mcpp'))

    logger.info("Fetching MC++ from %s and installing it to %s"
                % (url, downloader.destination()))

    downloader.get_zip_archive(url, dirOffset=1)

def main(argv):
    downloader = FileDownloader()
    downloader.parse_args(argv)
    get_mcpp(downloader)

if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    try:
        main(sys.argv[1:])
    except Exception as e:
        print(e.message)
        print("Usage: %s [--insecure] [target]" % os.path.basename(sys.argv[0]))
        raise
        sys.exit(1)

