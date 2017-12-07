"""
A daps SMPS reader. This will be built up slowly over time.
"""
from collections import OrderedDict
from pyomo.pysp.daps.distributions import UnivariateDiscrete

class SMPS_Indep_Discrete:
    """
    This class reads and "delivers" a sto file with indep discrete data.
    """
    def __init__(self, stofilename):
        """
        args: stofilename gives the file name with sto file data
        """
        def processoneRHS(RHS, stolines, cl):
            """
            local to init, update RHS using stolines starting at cl.
            Return cl ready for the next one.
            """
            bpoints = OrderedDict()
            parts = stolines[cl].split()
            name = lastname = parts[1]
            while lastname == name:
                parts = stolines[cl].split()
                if parts[0] == 'ENDATA':
                    cl += 1
                    break
                if parts[0] != "RHS":
                    raise RuntimeError(stofilename+": expecting only RHS")
                name = parts[1]
                if name != lastname:
                    break
                else:
                    lastname = name
                bpoints[float(parts[2])] = float(parts[3]) # val, prob
                cl += 1
            RHS[lastname] = UnivariateDiscrete(bpoints)
            return cl
            
        with open(stofilename) as f:
            inlines = f.readlines()
        inlines = [x.strip() for x in inlines]
        stolines = []
        for x in inlines:
            if x[0] != "*":
                stolines.append(x)
        if stolines[0][:5] != "STOCH":
            raise RuntimeError(stofilename+" does not begin with STOCH")
        parts = stolines[1].split()
        if len(parts) < 1 or parts[0] != "INDEP" or parts[1][:5] != "DISCR":
            raise RuntimeError(stofilename+": expecting INDEP      DISCRETE")
        if stolines[-1][:6] != "ENDATA":
            raise RuntimeError(stofilename+": expecting ENDATA to end file")
        self.RHS = {}  # dlw nov 2017: just do RHS now, add col later...
        cl = 3
        while cl < len(stolines):
            cl = processoneRHS(self.RHS, stolines, cl)

    def DrawOneSample(self):
        """
        return a dictionary with the RHS names as keys and a sample for each as the values
        """
        sample = {}
        for name in self.RHS:
            sample[name] = self.RHS[name].sample_one()
        return sample

if __name__ == '__main__':
    stoproc = SMPS_Indep_Discrete("ssn.sto")
    sample = stoproc.DrawOneSample()
    for name, val in sample.items():
        print (name, str(val))
        
