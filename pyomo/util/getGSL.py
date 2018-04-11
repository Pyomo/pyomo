import zipfile
import io
import os
import platform
import ssl
import stat
import sys
from six.moves.urllib.request import urlopen

# These URLs were retrieved from
#     https://ampl.com/resources/extended-function-library/
urlmap = {
    'linux':   'https://www.ampl.com/NEW/amplgsl/amplgsl.linux-intel%s.zip',
    'windows': 'https://www.ampl.com/NEW/amplgsl/amplgsl.mswin%s.zip',
    'cygwin':  'https://www.ampl.com/NEW/amplgsl/amplgsl.mswin%s.zip',
    'darwin':  'https://www.ampl.com/NEW/amplgsl/amplgsl.macosx%s.zip'
}

def get_gsl(fname=None, insecure=False):
    system = platform.system().lower()
    for c in '.-_':
        system = system.split(c)[0]
    bits = 64 if sys.maxsize > 2**32 else 32
    url = urlmap.get(system, None)
    if url is None:
        raise RuntimeError(
            "ERROR: cannot infer the correct url for platform '%s'" % platform)
    url = url % bits

    if fname is None:
        fname = '.'
    if os.path.isdir(fname):
        fname = os.path.join(fname, 'amplgsl.dll')

    print("Fetching GSL from %s and installing it to %s" % (url, fname))
    with open(fname, 'wb') as FILE:
        try:
            ctx = ssl.create_default_context()
            if insecure:
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
            fetch = urlopen(url, context=ctx)
        except AttributeError:
            # Revert to pre-2.7.9 syntax
            fetch = urlopen(url)
        zipped_file = io.BytesIO(fetch.read())
        print("  ...downloaded %s bytes" % (len(zipped_file.getvalue()),))
        raw_file = zipfile.ZipFile(zipped_file).open('amplgsl.dll').read()
        FILE.write(raw_file)
        print("  ...wrote %s bytes" % (len(raw_file),))

if __name__ == '__main__':
    if len(sys.argv) > 1 and '--insecure' in sys.argv:
        insecure = True
        sys.argv.remove('--insecure')
    else:
        insecure = False
    if len(sys.argv) > 1:
        fname = sys.argv.pop(1)
    else:
        fname = None
    if len(sys.argv) > 1:
        print("Usage: %s [--insecure] target" % sys.argv[0])
        sys.exit(1)
    get_gsl(fname=fname, insecure=insecure)

