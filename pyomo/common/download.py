#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import argparse
import gzip
import io
import logging
import os
import platform
import ssl
import sys
import zipfile

from six.moves.urllib.request import urlopen

from .config import PYOMO_CONFIG_DIR
from .errors import DeveloperError
import pyomo.common

logger = logging.getLogger('pyomo.common.download')

DownloadFactory = pyomo.common.Factory('library downloaders')

class FileDownloader(object):
    def __init__(self, insecure=False, cacert=None):
        self._fname = None
        self.target = None
        self.insecure = insecure
        self.cacert = cacert
        if cacert is not None:
            if not self.cacert or not os.path.isfile(self.cacert):
                raise RuntimeError(
                    "cacert='%s' does not refer to a valid file."
                    % (self.cacert,))


    def get_sysinfo(self):
        """Return a tuple (platform_name, bits) for the current system

        Returns
        -------
           platform_name (str): lower case, usually in {linux, windows,
              cygwin, darwin}.
           bits (int): OS address width in {32, 64}
        """

        system = platform.system().lower()
        for c in '.-_':
            system = system.split(c)[0]
        bits = 64 if sys.maxsize > 2**32 else 32
        return system, bits


    def get_url(self, urlmap):
        system, bits = self.get_sysinfo()
        url = urlmap.get(system, None)
        if url is None:
            raise RuntimeError(
                "cannot infer the correct url for platform '%s'"
                % (platform,))
        return url


    def create_parser(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()
        parser.add_argument(
            '--insecure',
            action='store_true',
            dest='insecure',
            default=False,
            help="Disable all SSL verification",
        )
        parser.add_argument(
            '--cacert',
            action='store',
            dest='cacert',
            default=None,
            help="Use CACERT as the file of certificate authorities "
            "to verify peers.",
        )
        return parser

    def parse_args(self, argv):
        parser = self.create_parser()
        parser.add_argument(
            'target',
            nargs="?",
            default=None,
            help="Target destination directory or filename"
        )
        parser.parse_args(argv, self)
        if self.cacert is not None:
            if not self.cacert or not os.path.isfile(self.cacert):
                raise RuntimeError(
                    "--cacert='%s' does not refer to a valid file."
                    % (self.cacert,))

    def set_destination_filename(self, default):
        if self.target is not None:
            self._fname = self.target
        else:
            self._fname = PYOMO_CONFIG_DIR
            if not os.path.isdir(self._fname):
                os.makedirs(self._fname)
        if os.path.isdir(self._fname):
            self._fname = os.path.join(self._fname, default)
        targetDir = os.path.dirname(self._fname)
        if not os.path.isdir(targetDir):
            os.makedirs(targetDir)

    def destination(self):
        return self._fname

    def retrieve_url(self, url):
        """Return the contents of a URL as an io.BytesIO object"""
        try:
            ctx = ssl.create_default_context(cafile=self.cacert)
            if self.insecure:
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
            fetch = urlopen(url, context=ctx)
        except AttributeError:
            # Revert to pre-2.7.9 syntax
            fetch = urlopen(url)
        ans = fetch.read()
        logger.info("  ...downloaded %s bytes" % (len(ans),))
        return ans


    def get_binary_file(self, url):
        if self._fname is None:
            raise DeveloperError("target file name has not been initialized "
                                 "with set_destination_filename")
        with open(self._fname, 'wb') as FILE:
            raw_file = self.retrieve_url(url)
            FILE.write(raw_file)
            logger.info("  ...wrote %s bytes" % (len(raw_file),))


    def get_binary_file_from_zip_archive(self, url, srcname):
        if self._fname is None:
            raise DeveloperError("target file name has not been initialized "
                                 "with set_destination_filename")
        with open(self._fname, 'wb') as FILE:
            zipped_file = io.BytesIO(self.retrieve_url(url))
            raw_file = zipfile.ZipFile(zipped_file).open(srcname).read()
            FILE.write(raw_file)
            logger.info("  ...wrote %s bytes" % (len(raw_file),))


    def get_zip_archive(self, url, dirOffset=0):
        if self._fname is None:
            raise DeveloperError("target file name has not been initialized "
                                 "with set_destination_filename")
        if os.path.exists(self._fname) and not os.path.isdir(self._fname):
            raise RuntimeError(
                "Target directory (%s) exists, but is not a directory"
                % (self._fname,))
        zip_file = zipfile.ZipFile(io.BytesIO(self.retrieve_url(url)))
        # Simple sanity checks
        names = []
        for info in zip_file.infolist():
            f = info.filename
            if f[0] in '\\/' or '..' in f:
                logger.error("malformed (potentially insecure) filename (%s) "
                             "found in zip archive.  Skipping file." % (f,))
                continue
            target = self._splitpath(f)
            if len(target) <= dirOffset:
                if f[-1] != '/':
                    logger.warning("Skipping file (%s) in zip archive due to "
                                   "dirOffset" % (f,))
                continue
            info.filename = target[-1] + '/' if f[-1] == '/' else target[-1]
            zip_file.extract(
                f, os.path.join(self._fname, *tuple(target[dirOffset:-1])))

    def get_gzipped_binary_file(self, url):
        if self._fname is None:
            raise DeveloperError("target file name has not been initialized "
                                 "with set_destination_filename")
        with open(self._fname, 'wb') as FILE:
            gzipped_file = io.BytesIO(self.retrieve_url(url))
            raw_file = gzip.GzipFile(fileobj=gzipped_file).read()
            FILE.write(raw_file)
            logger.info("  ...wrote %s bytes" % (len(raw_file),))


    def _splitpath(self, path):
        components = []
        head, tail = os.path.split(os.path.normpath(path))
        while head != path:
            if tail:
                components.append(tail)
            path = head
            head, tail = os.path.split(path)
        if head:
            components.append(head)
        components.reverse()
        return components
