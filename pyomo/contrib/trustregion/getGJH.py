import gzip
import io
import os
import platform
import ssl
import stat
import sys
from six.moves.urllib.request import urlopen

urlmap = {
    'linux':   'https://ampl.com/netlib/ampl/student/linux/gjh.gz',
    'windows': 'https://ampl.com/netlib/ampl/student/mswin/gjh.exe.gz',
    'cygwin':  'https://ampl.com/netlib/ampl/student/mswin/gjh.exe.gz',
    'darwin':  'https://ampl.com/netlib/ampl/student/macosx/x86_32/gjh.gz',
}

def get_gjh(fname=None, insecure=False):
    if fname is None:
        fname = 'gjh'

    system = platform.system().lower()
    for c in '.-_':
        system = system.split(c)[0]
    url = urlmap.get(system, None)
    if url is None:
        raise RuntimeError(
            "ERROR: cannot infer the correct url for platform '%s'" % platform)

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
        FILE.write(gzip.GzipFile(fileobj=gzipped_file).read())
    mode = os.stat(fname).st_mode
    os.chmod(fname, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--insecure':
        insecure = True
        sys.argv.pop(1)
    else:
        insecure = False
    if len(sys.argv) > 1:
        fname = sys.argv.pop(1)
        if os.path.isdir(fname):
            fname = os.path.join(fname, 'gjh')
    else:
        fname = None
    if len(sys.argv) > 1:
        print("Usage: %s [--insecure] target" % sys.argv[0])
        sys.exit(1)
    get_gjh(fname=fname, insecure=insecure)

