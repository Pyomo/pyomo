__all__ = ['OSSolverService', 'OSOptions']

import os.path
import sys
from pyomo.misc import pyomo_command


getAll = """<?xml version="1.0" encoding="UTF-8"?>
<ospl xmlns="os.optimizationservices.org">
  <processHeader>
     <request action="getAll"/>
  </processHeader>
  <processData/>
</ospl>"""


class Options(object):

    def __init__(self, name):
        self._name = name

    def __str__(self):
        tmp = '<'+self._name+'>'
        for item in dir(self):
            if item[0] != '_':
                print(self._name,item)
                if type(getattr(self,item)) is Options:
                    tmp += str(getattr(self,item))
                else:
                    tmp += '<'+item+'>'+str(getattr(self,item))+'</'+item+'>'
        tmp += '</'+self._name+'>'
        return tmp


class OSOptions(Options):

    def __init__(self):
        Options.__init__(self,'osol')
        self.general = Options('general')
        self.solver = Options('solver')


class OSSolverService(object):

    def __init__(self, url, verbose=False):
        self.verbose = verbose
        try:
            from suds.client import Client
        except:
            if self.verbose:
                print("WARNING: Failed to import from 'suds'")
            self.client = None
        else:
            if self.verbose:
                print("Connecting to url %s ..." % url)
                
            self.client = Client(url)
            if self.verbose:
                print("... done.")
        self.url = url
        print(self.client)

    def kill(self, osol=''):
        """Terminate a job on the server."""
        return self.client.service.kill(osol)

    def knock(self, ospl=getAll, osol=''):
        """Request process and job status information from the server in
        ospl format."""
        return self.client.service.knock(ospl, osol)

    def retrieve(self, osol=''):
        """Retrieve results from a solver.  This method
        requires a <jobID> element in the osol string."""
        return self.client.service.retrieve(osol)

    def send(self, osil, osol=''):
        """Perform asynchronous communication with a server.  This method
        requires a <jobID> element in the osol string.  It returns True if
        the problem was successfully submitted, and False otherwise."""
        return self.client.service.send(osil, osol)

    def solve(self, osil, osol='', osrl=None):
        """Perform synchronous communication with a server.  This method
        submits a model instance and waits for the solution."""
        if os.path.exists(osil):
            osil = ''.join(open(osil,'r').readlines())
        if os.path.exists(osol):
            osol = ''.join(open(osol,'r').readlines())
        results = self.client.service.solve(osil, osol)
        if not osrl is None:
            OUTPUT = open(osrl, 'w')
            OUTPUT.write(results+'\n')
            OUTPUT.close()
        else:
            return results

    def getJobID(self, osol=''):
        """Return a job ID that can be used to uniquely identify a
        job that is launched with the send() method."""
        return self.client.service.getJobID(osol)


@pyomo_command('PyomoOSSolverService', 'Launch an OS solver service')
def execute(argv=sys.argv):
    from optparse import OptionParser

    parser = OptionParser('OSSolverService [OPTIONS]')
    parser.add_option('--verbose', '-v', action='store_true', dest='verbose', default=False,
            help='Print debugging information.')
    parser.add_option('--osil', action='store', dest='osil', default=None,
            help='The name of a file that contains an optimization instance in OSiL format.')
    #parser.add_option('--nl', action='store', dest='nl', default=None,
    #        help='The name of a file that contains an optimization instance in NL format.')
    #parser.add_option('--mps', action='store', dest='mps', default=None,
    #        help='The name of a file that contains an optimization instance in MPS format.')
    parser.add_option('--osol', action='store', dest='osol', default='',
            help='The name of a file that contains solver options.')
    parser.add_option('--osrl', action='store', dest='osrl', default=None,
            help='The name of the file that will contain the solver results after optimization is performed.')
    parser.add_option('--solver', action='store', dest='solver', default=None,
            help='The name of the solver that will be used to optimize the problem.  The default value is cbc.')
    parser.add_option('--serviceLocation', '--url', action='store', dest='url', default=None,
            help='The url of the solver service.  If not specified, it is assumed that the problem is solved locally.')
    #parser.add_option('--methodName', '--method', action='store', dest='method', default=None,
    #        help='The method on the solver service that will be invoked.  Valid values are: solve (the default), send, kill, knock, getJobID, and retrieve.')
    #parser.add_option('--browser', action='store', dest='browser', default=None,
    #        help='A path to the browser on the local machine.  If this parameter is specified, then the solver result in OSrL format is transformed using XSLT into HTML and displayed in the browser.')
    #parser.add_option('--config', action='store', dest='config', default=None,
    #        help='A text files that contains values for input parameters.')

    try:
        import suds
    except:
        print("ERROR: The 'suds' package must be installed to run the OSSolverService.")
        sys.exit(-1)
    options, args = parser.parse_args(argv)

    if options.url is None:
        print("ERROR: Cannot execute solver service without a service url.\n\tUse the '--url' option to specify the solver service.")
        sys.exit(-1)
    if options.osil is None:
        print("ERROR: Must specify OSiL filename with the '--osil' option.")
        sys.exit(-1)

    service = OSSolverService(options.url, options.verbose)
    results = service.solve(options.osil, osol=options.osol)
    if not options.osrl is None:
        print("Writing results to file '%s'" % results)
        OUTPUT=open(options.osrl, 'w')
        OUTPUT.write(results+'\n')
        OUTPUT.close()
    else:
        print("")
        print("Solver Results:")
        print(results)
        print("")


def test():
    url = 'http://kipp.chicagobooth.edu/os/OSSolverService.jws?wsdl'
    service = OSSolverService(url)

    opt = OSOptions()
    opt.general.jobID = service.getJobID()
    print('1 '+service.send('parincLinear.osil',osol=opt))
    print('2 '+service.knock(osol=opt))
    print('3 '+service.retrieve(osol=opt))

if __name__ == '__main__':
    test()
