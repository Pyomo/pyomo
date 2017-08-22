"""
cutpoint_set.py

Temporary file for parsing cutpoints from a cutpoint file.
Files are going to have the following structure:
Cutpointvectors(cpts_names, name)
wide 0.0 widelow 0.1 widemid 0.9 widehigh 1.0
wideskew1 0.0 widesk1low 0.05 widesk1mid 0.85 widesk1high 1.0
wideskew2 0.0 widesk2low 0.1 widesk2mid 0.95 widesk2high 1.0
quick 0.0 quicklow 0.5 quickhigh 1.0

The cutpoint sets must immediately follow the line "Cutpointvectors(cpts_names, name)
"""

def parse_cutpoint_file(cutpoint_filename):
        cutpoint_sets = []
        start_reading = False
        with open(cutpoint_filename) as f:
            for line in f:
                line = _gobble_comment(line).strip()
                if not(line): # Line has nothing in it
                    continue

                if line == 'Cutpointvectors(cpts_names, name)':
                    start_reading = True

                if start_reading:
                    cutpoint_sets.append(CutpointSet(line))

        return cutpoint_sets


def _gobble_comment(line):
    comment_start = line.find('#')
    if comment_start != -1: # comment found in line
        return line[:comment_start]
    else:
        return line


class CutpointSet:
    def __init__(self, line):
        fields = line.split()
        self.name = fields[0]
        self.cutpoints = [float(cutpoint) for cutpoint in fields[1::2]]
        self.intervals = [(x1, x2) for x1, x2 in zip(self.cutpoints, self.cutpoints[1:])]
        self.cutpoint_names = fields[2::2]
        self.validate_fields()

        self.cutpoint_dictionary = {name: interval for name, interval in zip(self.cutpoint_names, self.intervals)}


    def get(self, key):
        return self.cutpoint_dictionary[key]

    def validate_fields(self):
        if self.cutpoints[0] != 0:
            raise RuntimeError("The first cutpoint in the set {} must be 0".format(self.name))

        if self.cutpoints[-1] != 1:
            raise RuntimeError("The last cutpoint in the set {} must be 1".format(self.name))

        # if not increasing
        if not all(x1 < x2 for x1, x2 in self.intervals):
            raise RuntimeError("The cutpoints in the set {} are not in order".format(self.name))