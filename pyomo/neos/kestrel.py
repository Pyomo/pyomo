#!/usr/bin/env python
#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#
# This software is a modified version of the Kestrel interface
# package that is provided by NEOS:  http://www.neos-server.org
#

import io
import os
import re
import six
import sys
import time
import socket
import gzip
import base64
import tempfile
import logging

class NEOS(object):
    # NEOS currently only supports HTTPS access
    scheme = 'https'
    host = 'neos-server.org'
    port = '3333'
    # Legacy NEOS HTTP interface
    #urlscheme = 'http'
    #port = '3332'


if sys.version_info[0] < 3:
    from urlparse import urlparse
    import httplib
    import xmlrpclib
    # ProxiedTransport from Python 2.x documentation
    # (https://docs.python.org/2/library/xmlrpclib.html)
    class ProxiedTransport(xmlrpclib.Transport):
        def set_proxy(self, proxy):
            self.proxy = urlparse(proxy)
            if not self.proxy.hostname:
                # User omitted scheme from the proxy; assume http
                self.proxy = urlparse('http://'+proxy)

        def make_connection(self, host):
            target = urlparse(host)
            if target.scheme:
                self.realhost = target.geturl()
            else:
                self.realhost = '%s://%s' % (NEOS.scheme, target.geturl())

            # Empirically, the connection class in Python 2.7 needs to
            # match the PROXY connection scheme, and the final endpoint
            # scheme needs to be specified in the POST below.
            if self.proxy.scheme == 'https':
                connClass = httplib.HTTPSConnection
            else:
                connClass = httplib.HTTPConnection
            return connClass(self.proxy.hostname, self.proxy.port)


        def send_request(self, connection, handler, request_body):
            connection.putrequest(
                "POST", '%s%s' % (self.realhost, handler))

        def send_host(self, connection, host):
            connection.putheader('Host', self.realhost)

else: # Python 3.x
    from urllib.parse import urlparse
    import http.client as httplib
    import xmlrpc.client as xmlrpclib
    # ProxiedTransport from Python 3.x documentation
    # (https://docs.python.org/3/library/xmlrpc.client.html)
    class ProxiedTransport(xmlrpclib.Transport):
        def set_proxy(self, host):
            self.proxy = urlparse(host)
            if not self.proxy.hostname:
                # User omitted scheme from the proxy; assume http
                self.proxy = urlparse('http://'+proxy)

        def make_connection(self, host):
            scheme = urlparse(host).scheme
            if not scheme:
                scheme = NEOS.scheme

            # Empirically, the connection class in Python 3.x needs to
            # match the final endpoint connection scheme, NOT the proxy
            # scheme.  The set_tunnel host then should NOT have a scheme
            # attached to it.
            if scheme == 'https':
                connClass = httplib.HTTPSConnection
            else:
                connClass = httplib.HTTPConnection

            connection = connClass(self.proxy.hostname, self.proxy.port)
            connection.set_tunnel(host)
            return connection

logger = logging.getLogger('pyomo.solvers')


class kestrelAMPL:

    def __init__(self):
        self.setup_connection()

    def setup_connection(self):
        # on *NIX, the proxy can show up either upper or lowercase.
        # Prefer lower case, and prefer HTTPS over HTTP if the
        # NEOS.scheme is https.
        proxy = os.environ.get(
            'http_proxy', os.environ.get(
                'HTTP_PROXY', ''))
        if NEOS.scheme == 'https':
            proxy = os.environ.get(
                'https_proxy', os.environ.get(
                    'HTTPS_PROXY', proxy))
        transport = None
        if proxy:
            transport = ProxiedTransport()
            transport.set_proxy(proxy)

        self.neos = xmlrpclib.ServerProxy(
            "%s://%s:%s" % (NEOS.scheme, NEOS.host, NEOS.port),
            transport=transport)

        logger.info("Connecting to the NEOS server ... ")
        try:
            result = self.neos.ping()
            logger.info("OK.")
        except socket.error:
            e = sys.exc_info()[1]
            self.neos = None
            logger.info("Fail.")
            logger.warning("NEOS is temporarily unavailable.\n")

    def tempfile(self):
        return os.path.join(tempfile.gettempdir(),'at%s.jobs' % os.getenv('ampl_id'))

    def kill(self,jobnumber,password):
        response = self.neos.killJob(jobNumber,password)
        sys.stdout.write(response+"\n")

    def solvers(self):
        return self.neos.listSolversInCategory("kestrel") \
                if not self.neos is None else []

    def retrieve(self,stub,jobNumber,password):
        # NEOS should return results as uu-encoded xmlrpclib.Binary data
        results = self.neos.getFinalResults(jobNumber,password)
        if isinstance(results,xmlrpclib.Binary):
            results = results.data
        #decode results to kestrel.sol
        # Well try to anyway, any errors will result in error strings in .sol file
        #  instead of solution.
        if stub[-4:] == '.sol':
            stub = stub[:-4]
        solfile = open(stub + ".sol","wb")
        solfile.write(results)
        solfile.close()

    def submit(self, xml):
        user = "%s on %s" % (os.getenv('LOGNAME'),socket.getfqdn(socket.gethostname()))
        (jobNumber,password) = self.neos.submitJob(xml,user,"kestrel")
        if jobNumber == 0:
            sys.stdout.write("Error: %s\nJob not submitted.\n" % password)
            sys.exit(1)
        sys.stdout.write("Job %d submitted to NEOS, password='%s'\n" %
                         (jobNumber,password))
        sys.stdout.write("Check the following URL for progress report :\n")
        sys.stdout.write(urlscheme+"://www.neos-server.org/neos/cgi-bin/nph-neos-solver.cgi?admin=results&jobnumber=%d&pass=%s\n" % (jobNumber,password))
        return (jobNumber,password)

    def getJobAndPassword(self):
        """
        If kestrel_options is set to job/password, then return
        the job and password values
        """
        jobNumber=0
        password=""
        options = os.getenv("kestrel_options")
        if options is not None:
            m = re.search(r'job\s*=\s*(\d+)',options,re.IGNORECASE)
            if m:
                jobNumber = int(m.groups()[0])
            m = re.search(r'password\s*=\s*(\S+)',options,re.IGNORECASE)
            if m:
                password = m.groups()[0]
        return (jobNumber,password)

    def getSolverName(self):
        """
        Read in the kestrel_options to pick out the solver name.
        The tricky parts:
          we don't want to be case sensitive, but NEOS is.
          we need to read in options variable
        """
        # Get a list of available kestrel solvers from NEOS
        allKestrelSolvers = self.neos.listSolversInCategory("kestrel")
        kestrelAmplSolvers = []
        for s in allKestrelSolvers:
            i = s.find(':AMPL')
            if i > 0:
                kestrelAmplSolvers.append(s[0:i])
        self.options = None
        # Read kestrel_options to get solver name
        if "kestrel_options" in os.environ:
            self.options = os.getenv("kestrel_options")
        elif "KESTREL_OPTIONS" in os.environ:
            self.options = os.getenv("KESTREL_OPTIONS")
        #
        if self.options is not None:
            m = re.search('solver\s*=*\s*(\S+)',self.options,re.IGNORECASE)
            NEOS_solver_name=None
            if m:
                solver_name=m.groups()[0]
                for s in kestrelAmplSolvers:
                    if s.upper() == solver_name.upper():
                      NEOS_solver_name=s
                      break
                #
                if not NEOS_solver_name:
                    sys.stdout.write("%s is not available on NEOS.  Choose from:\n" % solver_name)
                    for s in kestrelAmplSolvers:
                        sys.stdout.write("\t%s\n"%s)
                    sys.stdout.write('To choose: option kestrel_options "solver=xxx";\n\n')
                    sys.exit(1)
        #
        if self.options is None or m is None:
            sys.stdout.write("No solver name selected.  Choose from:\n")
            for s in kestrelAmplSolvers:
                sys.stdout.write("\t%s\n"%s)
            sys.stdout.write('\nTo choose: option kestrel_options "solver=xxx";\n\n')
            sys.exit(1)
        return NEOS_solver_name

    def formXML(self,stub):
        solver = self.getSolverName()
        zipped_nl_file = io.BytesIO()
        if os.path.exists(stub) and stub[-3:] == '.nl':
            stub = stub[:-3]
        nlfile = open(stub+".nl","rb")
        zipper = gzip.GzipFile(mode='wb',fileobj=zipped_nl_file)
        zipper.write(nlfile.read())
        zipper.close()
        nlfile.close()
        #
        ampl_files={}
        for key in ['adj','col','env','fix','spc','row','slc','unv']:
            if os.access(stub+"."+key,os.R_OK):
                f = open(stub+"." +key,"r")
                val=""
                buf = f.read()
                while buf:
                    val += buf
                    buf=f.read()
                f.close()
                ampl_files[key] = val
        # Get priority
        priority = ""
        m = re.search(r'priority[\s=]+(\S+)',self.options)
        if m:
            priority = "<priority>%s</priority>\n" % (m.groups()[0])
        # Add any AMPL-created environment variables to dictionary
        solver_options = "kestrel_options:solver=%s\n" % solver.lower()
        solver_options_key = "%s_options" % solver
        #
        solver_options_value = ""
        if solver_options_key in os.environ:
            solver_options_value = os.getenv(solver_options_key)
        elif solver_options_key.lower() in os.environ:
            solver_options_value = os.getenv(solver_options_key.lower())
        elif solver_options_key.upper() in os.environ:
            solver_options_value = os.getenv(solver_options_key.upper())
        if not solver_options_value == "":
            solver_options += "%s_options:%s\n" % (solver.lower(), solver_options_value)
        #
        if six.PY2:
            nl_string = base64.encodestring(zipped_nl_file.getvalue())
        else:
            nl_string = (base64.encodestring(zipped_nl_file.getvalue())).decode('utf-8')
        xml = """
              <document>
              <category>kestrel</category>
              <solver>%s</solver>
              <inputType>AMPL</inputType>
              %s
              <solver_options>%s</solver_options>
              <nlfile><base64>%s</base64></nlfile>\n""" %\
                                (solver,priority,
                                 solver_options,
                                 nl_string)
        #
        for key in ampl_files:
            xml += "<%s><![CDATA[%s]]></%s>\n" % (key,ampl_files[key],key)
        #
        for option in ["kestrel_auxfiles","mip_priorities","objective_precision"]:
            if option in os.environ:
                xml += "<%s><![CDATA[%s]]></%s>\n" % (option,os.getenv(option),option)
        #
        xml += "</document>"
        return xml



if __name__=="__main__":            #pragma:nocover
  if len(sys.argv) < 2:
    sys.stdout.write("kestrel should be called from inside AMPL.\n")
    sys.exit(1)

  kestrel = kestrelAMPL()

  if sys.argv[1] == "solvers":
    for s in sorted(kestrel.neos.listSolversInCategory("kestrel")):
        print(" "+s)
    sys.exit(0)

  elif sys.argv[1] == "submit":
    xml = kestrel.formXML("kestproblem")
    (jobNumber,password) = kestrel.submit(xml)


    # Add the job,pass to the stack
    jobfile = open(kestrel.tempfile(),'a')
    jobfile.write("%d %s\n" % (jobNumber,password))
    jobfile.close()

  elif sys.argv[1] == "retrieve":
    # Pop job,pass from the stack
    try:
      jobfile = open(kestrel.tempfile(),'r')
    except IOError:
      e = sys.exc_info()[1]
      sys.stdout.write("Error, could not open file %s.\n")
      sys.stdout.write("Did you use kestrelsub?\n")
      sys.exit(1)

    m = re.match(r'(\d+) ([a-zA-Z]+)',jobfile.readline())
    if m:
      jobNumber = int(m.groups()[0])
      password = m.groups()[1]
    restofstack = jobfile.read()
    jobfile.close()

    kestrel.retrieve('kestresult',jobNumber,password)

    if restofstack:
      sys.stdout.write("restofstack: %s\n" % restofstack)
      jobfile = open(kestrel.tempfile(),'w')
      jobfile.write(restofstack)
      jobfile.close()
    else:
      os.unlink(kestrel.tempfile())

  elif sys.argv[1] == "kill":
    (jobNumber,password) = kestrel.getJobAndPassword()
    if jobNumber:
      kestrel.kill(jobNumber,password)
    else:
      sys.stdout.write("To kill a NEOS job, first set kestrel_options variable:\n")
      sys.stdout.write('\tampl: option kestrel_options "job=#### password=xxxx";\n')
  else:
    try:
      stub = sys.argv[1]
      # See if kestrel_options has job=.. password=..
      (jobNumber,password) = kestrel.getJobAndPassword()

      # otherwise, submit current problem to NEOS
      if not jobNumber:
        xml = kestrel.formXML(stub)
        (jobNumber,password) = kestrel.submit(xml)
      
    except KeyboardInterrupt:
      e = sys.exc_info()[1]
      sys.stdout.write("Keyboard Interrupt while submitting problem.\n")
      sys.exit(1)
    try:
      # Get intermediate results
      time.sleep(1)
      status = "Running"
      offset = 0
      while status == "Running" or status == "Waiting":
        (output,offset) = kestrel.neos.getIntermediateResults(jobNumber,
                                                           password,offset)
      
        if isinstance(output,xmlrpclib.Binary):
          output = output.data
        sys.stdout.write(output)
        status = kestrel.neos.getJobStatus(jobNumber,password)
        time.sleep(5)

      # Get final results
      kestrel.retrieve(stub,jobNumber,password)
      sys.exit(0)
    except KeyboardInterrupt:
      e = sys.exc_info()[1]
      msg = '''
Keyboard Interrupt\n\
Job is still running on remote machine\n\
To stop job:\n\
\tampl: option kestrel_options "job=%d password=%s";\n\
\tampl: commands kestrelkill;\n\
To retrieve results:\n\
\tampl: option kestrel_options "job=%d password=%s";\n\
\tampl: solve;\n''' % (jobNumber,password,jobNumber,password)
      sys.stdout.write(msg)
      sys.exit(1)
