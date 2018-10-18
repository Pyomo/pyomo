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
import stat
import sys
from pyomo.common.download import FileDownloader

logger = logging.getLogger('pyomo.common')

# These URLs were retrieved from
#     https://ampl.com/resources/hooking-your-solver-to-ampl/
urlmap = {
    'linux':   'https://ampl.com/netlib/ampl/student/linux/gjh.gz',
    'windows': 'https://ampl.com/netlib/ampl/student/mswin/gjh.exe.gz',
    'cygwin':  'https://ampl.com/netlib/ampl/student/mswin/gjh.exe.gz',
    'darwin':  'https://ampl.com/netlib/ampl/student/macosx/x86_32/gjh.gz',
}
exemap = {
    'linux':   '',
    'windows': '.exe',
    'cygwin':  '.exe',
    'darwin':  '',
}

def get_gjh(downloader=None):
    if downloader is None:
        downloader = FileDownloader()

    system, bits = downloader.get_sysinfo()
    url = downloader.get_url(urlmap)

    downloader.resolve_filename('gjh'+exemap[system])

    logger.info("Fetching GJH from %s and installing it to %s"
                % (url, downloader.fname))

    downloader.get_gzipped_binary_file(url)

    mode = os.stat(downloader.fname).st_mode
    os.chmod( downloader.fname,
              mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH )

def main(argv):
    downloader = FileDownloader()
    downloader.parse_args(argv)
    get_gjh(downloader)

if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    try:
        main(sys.argv[1:])
    except Exception as e:
        print(e.message)
        print("Usage: %s [--insecure] [target]" % os.path.basename(sys.argv[0]))
        sys.exit(1)

