import gzip
import io
import os
import platform
import ssl
import stat
import sys
from six.moves.urllib.request import urlopen

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

def get_gjh(fname=None, insecure=False):
    system = platform.system().lower()
    for c in '.-_':
        system = system.split(c)[0]
    url = urlmap.get(system, None)
    if url is None:
        raise RuntimeError(
            "ERROR: cannot infer the correct url for platform '%s'" % platform)

    if fname is None:
        fname = '.'
    if os.path.isdir(fname):
        fname = os.path.join(fname, 'gjh'+exemap[system])

    print("Fetching GJH from %s and installing it to %s" % (url, fname))
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
        gzipped_file = io.BytesIO(fetch.read())
        print("  ...downloaded %s bytes" % (len(gzipped_file.getvalue()),))
        raw_file = gzip.GzipFile(fileobj=gzipped_file).read()
        FILE.write(raw_file)
        print("  ...wrote %s bytes" % (len(raw_file),))
    mode = os.stat(fname).st_mode
    os.chmod(fname, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

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
    get_gjh(fname=fname, insecure=insecure)

