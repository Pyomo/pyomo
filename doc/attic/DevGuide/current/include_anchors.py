import re
import sys
import os.path
import os

separator = '@'

#include = re.compile('include::([\w\.\-/\/]+\_[a-zA-Z0-9]*\.[a-zA-Z0-9])\[\]')
include = re.compile('include::(.+%s[^%s\.]*\.[^\.]+)\[\]' % (separator,separator))
split_ref = re.compile('(.+)%s([^%s\.]*)\.([^\.]+)$' % (separator,separator))
anchor_start = re.compile('# @([a-zA-Z0-9]+):')
anchor_end = re.compile('# @:([a-zA-Z0-9]+)')

processed = set()

def open_source(target):
    m = split_ref.match(target)
    if not m:
        print "ERROR: target \"%s\" is not a valid target name" % (target,)
        sys.exit(1)
    root, anchor, ext = m.groups()
    global processed
    if root in processed:
        return
    #
    src_fname = '%s.%s' % (root, ext)
    if not os.path.exists(src_fname):
        print "ERROR: source file '%s' does not exist!" % src_fname
        sys.exit(1)
    INPUT = open(src_fname, 'r')
    return INPUT, root, ext, anchor

def generate_all(target):
    #
    INPUT, root, ext = open_source(target)[:3]
    anchors = {}
    stripped_src = '%s%s.%s' % (root, separator, ext)
    anchors[''] = open(stripped_src, 'w')
    print "Generating '%s' ..." % stripped_src
    for line in INPUT:
        m2 = pat2.match(line)
        m3 = pat3.match(line)
        if m2:
            anchor = m2.group(1)
            fname = '%s%s%s.%s' % (root, separator, anchor, ext)
            anchors[anchor] = open(fname, 'w')
            print "Generating '%s' ..." % fname
        elif m3:
            anchor = m3.group(1)
            anchors[anchor].close()
            del anchors[anchor]
        else:
            for anchor in anchors:
                #os.write(anchors[anchor].fileno(), line)
                anchors[anchor].write(line)
    INPUT.close()
    for anchor in anchors:
        if anchor != '':
            print "ERROR: anchor '%s' did not terminate" % anchor
        anchors[anchor].close()
    #
    global processed
    processed.add(root)


def generate_file(target):
    INPUT, root, ext, anchor = open_source(target)
    OUTPUT = open('%s%s%s.%s' % (root, separator, anchor, ext), 'w')
    outputting = anchor == ''
    for line in INPUT:
        m1 = anchor_start.match(line)
        m2 = anchor_end.match(line)
        if m1:
            if m1.group(1) == anchor:
                outputting = True
        elif m2:
            if m2.group(1) == anchor:
                outputting = False
        elif outputting:
            OUTPUT.write(line)
    if outputting and anchor != '':
        print "ERROR: anchor '%s' did not terminate" % anchor
    INPUT.close()
    OUTPUT.close()
    

all = True
for file in sys.argv[1:]:
    if file == '--':
        all = False
        continue

    if all:
        print "Processing all references in file '%s' ..." % file
        INPUT = open(file, 'r')
        for line in INPUT:
            m = include.match(line)
            if m:
                generate_all(m.group(1))
        INPUT.close()
    else:
        print "Extracting file '%s' ..." % file
        generate_file(file)

