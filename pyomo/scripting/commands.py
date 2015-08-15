#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import code
import sys
import os
import string
import signal
import subprocess
import pyutilib.subprocess

import pyutilib.pyro
from pyutilib.misc import Options
from pyomo.opt import SolverResults
from pyomo.util._command import pyomo_command
import pyomo.scripting.pyomo_parser

@pyomo_command('pyomo_ns', "Launch a Pyro name server for Pyomo")
def pyomo_ns():
    pyutilib.pyro.start_ns()

@pyomo_command('pyomo_nsc', "Execute the Pyro name server control tool for Pyomo")
def pyomo_nsc():
    pyutilib.pyro.start_nsc()

@pyomo_command('kill_pyro_mip_servers', "Terminate Pyomo's MIP solvers using Pyro")
def kill_pyro_mip_servers():
    if len(sys.argv) > 2:
        print("***Incorrect invocation - use: kill_pyro_mip_servers pid-filename")
        sys.exit(1)

    pid_filename = "pyro_mip_servers.pids"
    if len(sys.argv) == 2:
        pid_filename = sys.argv[1]

    print("Killing pyro mip servers specified in file="+pid_filename)

    pid_file = open(pid_filename, "r")
    for line in pid_file.readlines():
        pid = eval(string.strip(line))
        print("KILLING PID="+str(pid))
        os.kill(pid, signal.SIGTERM)
    pid_file.close()

@pyomo_command('launch_pyro_mip_servers', "Launch Pyomo's MIP solvers using Pyro")
def launch_pyro_mip_servers():
    if len(sys.argv) != 2:
        print("***Incorrect invocation - use: launch_pyro_mip_servers num-servers")
        sys.exit(1)

    num_servers = eval(sys.argv[1])

    print("Number of servers to launch="+str(num_servers))

    server_pids = []

    for i in range(1, num_servers+1):
        print("Launching server number "+str(i))
        output_filename = "pyro_mip_server"+str(i)+".out"
        # the "exec" ensures that (at least for bash) that the server process
        # will be the process returned, i.e., it becomes the child process - no
        # shell process intermediate. more correctly, exec exits the current
        # process before it does so (no fork).
        pid=subprocess.Popen("exec pyro_mip_server >& pyro_mip_server."+str(i)+".out", shell=True).pid
        server_pids.append(pid)

    # perhaps a better place would be in the users home directory, but I'll
    # worry about that a bit later.
    pid_output_filename = "pyro_mip_servers.pids"
    pid_output_file = open(pid_output_filename,"w")
    for pid in server_pids:
        pid_output_file.write(str(pid)+'\n')
    pid_output_file.close()

    print("PIDs for launched servers recorded in file="+pid_output_filename)

@pyomo_command('OSSolverService', "Launch an OS solver service")
def OSSolverService():
    import pyomo.opt

    if len(sys.argv) == 1:
        print("OSSolverService -osil <filename> -solver <name>")
        sys.exit(1)

    osilFile = None
    solver = None
    i=1
    while i<len(sys.argv):
        if sys.argv[i] == "-osil":
            i=i+1
            osilFile=sys.argv[i]
        elif sys.argv[i] == "-solver":
            i=i+1
            solver=sys.argv[i]
        i=i+1

    print("osilFile",osilFile,"solver",solver)

    with pyomo.opt.SolverFactory(solver) as opt:
        opt.solve(osilFile, rformat=pyomo.opt.ResultsFormat.osrl)

@pyomo_command('readsol', "Read and print a *.sol file")
def readsol():
    import pyomo.opt

    reader = pyomo.opt.ReaderFactory("sol")
    soln = reader(sys.argv[1])
    soln.write()

@pyomo_command('pyomo_python', "Launch script using Pyomo's python installation")
def pyomo_python(args=None):
    if args is None:
        args = sys.argv[1:]
    if args is None or len(args) == 0:
        console = code.InteractiveConsole()
        console.interact('Pyomo Python Console\n'+sys.version)
    else:
        cmd = sys.executable+' '+ ' '.join(args)
        pyutilib.subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)

@pyomo_command('pyomo', "The main command interface for Pyomo")
def pyomo(args=None):
    parser = pyomo.scripting.pyomo_parser.get_parser()
    if args is None:
        ret = parser.parse_args()
    else:
        ret = parser.parse_args(args)
    ret.func(ret)


@pyomo_command('results_schema', "Print the predefined schema for a results object")
def results_schema():
    if len(sys.argv) > 1:
        print("results_schema  - Print the predefined schema in a SolverResults object")
    options = Options(schema=True)
    r=SolverResults()
    repn = r._repn_(options)
    r.pprint(sys.stdout, options, repn=repn)

