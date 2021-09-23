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
import glob
import os
import stat
import sys
from pyomo.common.download import FileDownloader

logger = logging.getLogger('pyomo.common')

# These URLs were retrieved from
#     https://ampl.com/resources/hooking-your-solver-to-ampl/
# All 32-bit downloads are used - 64-bit is available only for Linux
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

def readgjh(fname=None):
    """
    Build objective gradient and constraint Jacobian
    from gjh file written by the ASL gjh 'solver'.

    gjh solver may be called using pyomo and 'keepfiles=True'.
    Enable 'symbolic_solver_labels' as well to write
    .row and col file to get variable mappings.

    Parameters
    ----------
    fname : string, optional
        gjh file name. The default is None.

    Returns
    -------
    g : list
        Current objective gradient.
    J : list
        Current objective Jacobian.
    H : list
        Current objective Hessian.
    variableList : list
        Variables as defined by *.col file.
    constraintList : list
        Constraints as defined by *.row file.

    """
    if fname is None:
        files = list(glob.glob("*.gjh"))
        fname = files.pop(0)
        if len(files) > 1:
            print("**** WARNING **** More than one gjh file in current directory")
            print("  Processing: %s\nIgnoring: %s" % (
                fname, '\n         '.join(files)))

    with open(fname, "r") as f:
        line = "dummy_str"
        # Skip top lines until g value is reached
        while line != "param g :=\n":
            line = f.readline()

        line = f.readline()
        g = []
        # ; is the escape character
        while line[0] != ';':
            """
            When printed via ampl interface:
            ampl: display g;
            g [*] :=
                	 1   0.204082
                	 2   0.367347
                	 3   0.44898
                	 4   0.44898
                	 5   0.244898
                	 6  -0.173133
                	 7  -0.173133
                	 8  -0.0692532
                	 9   0.0692532
                	10   0.346266
                	;
            """
            index = int(line.split()[0]) - 1
            value = float(line.split()[1])
            g.append([index, value])
            line = f.readline()

        # Skip lines until J value is reached 
        while line != "param J :=\n":
            line = f.readline()

        line = f.readline()
        J = []
        while line[0] != ';':
            """
            When printed via ampl interface:
            ampl: display J;
            J [*,*]
           	:         1             2           3          4           5            6
           	 :=
           	1    -0.434327       0.784302     .          .           .          -0.399833
           	2     2.22045e-16     .          1.46939     .           .          -0.831038
           	3     0.979592        .           .         1.95918      .          -0.9596
           	4     1.79592         .           .          .          2.12245     -0.692532
           	5     0.979592        .           .          .           .           0
           	6      .            -0.0640498   0.545265    .           .            .
           	7      .             0.653061     .         1.14286      .            .
           	8      .             1.63265      .          .          1.63265       .
           	9      .             1.63265      .          .           .            .
           	10     .              .          0.262481   0.262481     .            .
           	11     .              .          1.14286     .          0.653061      .
           	12     .              .          1.95918     .           .            .
           	13     .              .           .         0.545265   -0.0640498     .
           	14     .              .           .         1.95918      .            .
           	15     .              .           .          .          1.63265       .
           	16     .              .           .          .           .          -1
           
           	:        7           8           9          10       :=
           	1     0.399833     .           .          .
           	2      .          0.831038     .          .
           	3      .           .          0.9596      .
           	4      .           .           .         0.692532
           	6    -0.799667    0.799667     .          .
           	7    -1.38506      .          1.38506     .
           	8    -1.33278      .           .         1.33278
           	9     0            .           .          .
           	10     .         -0.9596      0.9596      .
           	11     .         -1.38506      .         1.38506
           	12     .          0            .          .
           	13     .           .         -0.799667   0.799667
           	14     .           .          0           .
           	15     .           .           .         0
           	16    1            .           .          .
           	17   -1           1            .          .
           	18     .         -1           1           .
           	19     .           .         -1          1
            ;
            """
            if line[0] == '[':
                # Jacobian row index
                row = int(''.join(filter(str.isdigit, line))) - 1
                line = f.readline()

            column = int(line.split()[0]) - 1
            value = float(line.split()[1])
            J.append([row, column, value])
            line = f.readline()

        while line != "param H :=\n":
            line = f.readline()

        line = f.readline()
        H = []
        while line[0] != ';':
            """
            ampl: display H;
            	H [*,*]
            	:       1           2           3           4           5           6         :=
            	1      .         0.25         .           .           .         -0.35348
            	2     0.25        .          0.25         .           .         -0.212088
            	3      .         0.25         .          0.25         .           .
            	4      .          .          0.25         .          0.25         .
            	5      .          .           .          0.25         .           .
            	6    -0.35348   -0.212088     .           .           .         -0.0999584
            	7     0.35348   -0.212088   -0.35348      .           .          0.0999584
            	8      .         0.424176   -0.070696   -0.424176     .           .
            	9      .          .          0.424176    0.070696   -0.424176     .
            	10     .          .           .          0.35348     0.424176     .
            
            	:        7            8           9          10        :=
            	1     0.35348       .           .           .
            	2    -0.212088     0.424176     .           .
            	3    -0.35348     -0.070696    0.424176     .
            	4      .          -0.424176    0.070696    0.35348
            	5      .            .         -0.424176    0.424176
            	6     0.0999584     .           .           .
            	7    -0.299875     0.199917     .           .
            	8     0.199917    -0.439817    0.2399       .
            	9      .           0.2399     -0.439817    0.199917
            	10     .            .          0.199917   -0.199917
            	;
            """
            if line[0] == '[':
                # Hessian row index
                row = int(''.join(filter(str.isdigit, line))) - 1
                line = f.readline()

            column = int(line.split()[0]) - 1
            value = float(line.split()[1])
            H.append([row, column, value])
            line = f.readline()
    
    with open(fname[:-3]+'col', 'r') as f:
        data = f.read()
        variableList = data.split()

    with open(fname[:-3]+'row', 'r') as f:
        data = f.read()
        constraintList = data.split()

    return g, J, H, variableList, constraintList


def get_gjh(downloader):
    system, bits = downloader.get_sysinfo()
    url = downloader.get_platform_url(urlmap)

    downloader.set_destination_filename(
        os.path.join('bin', 'gjh'+exemap[system]))

    logger.info("Fetching GJH from %s and installing it to %s"
                % (url, downloader.destination()))

    downloader.get_gzipped_binary_file(url)

    mode = os.stat(downloader.destination()).st_mode
    os.chmod( downloader.destination(),
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
