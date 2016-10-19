# Author: David L. Woodruff October 2016
# 
# This python script creates scripts (e.g., slurm scripts) to run ph.
# I use this to avoid inconsistencies because bash variables
# cannot be used within slurm directives and also to be able
# to create "the same" script for multiple types of machines.
# Also, comments in slurm scripts can cause trouble if they are
# too early in the script, but I can put comments in this py file.

import datetime
import os

### Edit the following input lines:

#MachineOnWhichToRun = "NERSCslurm"
MachineOnWhichToRun = "sharedmemory"
workercnt = 5
modelname = "models"
instance = "1ef50"  # the runph line has "-i "+instance
ScriptFileName = instance+".bash"
nsport = 9926  # None to omit
bcport = 9092  # None to omit
threadcnt = 2 # number of threads to each subproblem

# distributed memory stuff
timelimit = "00:01:00"
nodecnt = 1
combined_srun = True
dispatch_memory = 10000  # at NERSC these are MB
worker_memory = 50000
ph_memory = 10000

# NERSC stuff
partition = "debug"

# Ugly runph line: runph -i instance -m modelname is assumed.
# Note that \\\n outputs a bash continuation line symbol and eoln
# Note that the threads substitution is done here.
runph_line = \
    "--rho-cfgfile=config/rhosetter05 \\\n" + \
    "--enable-ww-extensions \\\n" + \
    "--ww-extension-cfgfile=config/wwph-fixlag10.cfg  \\\n" + \
    "--ww-extension-suffixfile=config/wwph.suffixes \\\n" + \
    "--default-rho=1 \\\n" +\
    "--user-defined-extension=pyomo.pysp.plugins.phboundextension \\\n" + \
    "--traceback  \\\n" + \
    "--xhat-method=config/xhatslam.yaml \\\n" +\
    "--solver-manager=phpyro --shutdown-pyro \\\n" +\
    "--max-iterations=1 \\\n" +\
    '--scenario-solver-options="threads='+str(threadcnt)+'" \\\n' +\
    "--output-times  \\\n" + \
    " >& runph.out"

    # the threads line should ouput something like:
    #    --scenario-solver-options="threads=8" \


# end input lines ###

timestamp = datetime.datetime.now()
thisfile = os.path.abspath(__file__)

with open(ScriptFileName ,"wt") as f:
    
    if MachineOnWhichToRun == "NERSCslurm":
        f.write("#!/bin/bash -l\n")
        f.write("\n")
        f.write("#SBATCH --partition="+partition+"\n")
        f.write("#SBATCH --nodes="+str(nodecnt)+"\n")
        f.write("#SBATCH --time="+timelimit+"\n")
        f.write("#SBATCH --job-name="+instance+"\n")
        f.write("#SBATCH --license=SCRATCH\n")
        f.write("#SBATCH -A m2528\n")
        f.write("#SBATCH --output="+instance+".out\n")
        f.write("#SBATCH --gres=craynetwork:4\n")
        f.write("\n")
        f.write("module load python/3.5-anaconda\n")
        f.write("source activate pyomo\n")
        f.write("\n")
        # the plus 3 is for the name server, dispatch server and ph
        f.write("export PYRO_THREADPOOL_SIZE="+str(workercnt+3)+"\n")
        f.write("\n")
        f.write("export PYRO_HOST=$(hostname)\n")
        f.write("export PYRO_NS_HOST=$PYRO_HOST\n")
        f.write("export PYRO_NS_HOSTNAME=$PYRO_HOST\n")
        f.write("export PYRO_NS_BCHOST=$PYRO_HOST\n")
        f.write("\n")
        if nsport is not None:
            f.write("export PYRO_NS_PORT="+str(nsport)+"\n")
        if bcport is not None:
            f.write("export PYRO_NS_BC_PORT="+str(bcport)+"\n")
        f.write("\n")
        f.write("export PYRO_SOCK_REUSE=True\n")
        f.write("export PYRO_BROKEN_MSGWAITALL=1\n")
        f.write("export PYTHONDONTWRITEBYTECODE=1\n")
        f.write("\n")
        f.write("srun -u --relative=0 -N 1 --mem=1000 " + \
                " --gres=craynetwork:1 pyomo_ns " + \
                ">& pyomo_ns.out &\n")
        f.write("srun -u --relative=0 -N 1 --mem=" + str(dispatch_memory) + \
              "--gres=craynetwork:1 dispatch_srvr >& dispatch_srvr.out &\n")

        if combined_srun:
            # note that the output is going to all go to the same file
            f.write("srun -u -n "+str(workercnt)+" --mem="+str(worker_memory) + \
                    " --gres=craynetwork:1  " + \
                    "phsolverserver --verbose --traceback &\n")
        else:
            for w in range(workercnt):
                f.write("srun -u  -N 1  --gres=craynetwork:1 phsolverserver " + \
                        "--verbose --traceback >& worker"+str(w+1)+".out &\n")

        # note that ph command line does not terminate with an ampersand
        f.write ("srun -u --relative=0 -N 1 --mem="+str(ph_memory) + \
                 "--gres=craynetwork:1 runph -i "+instance + " -m "+modelname+" "+\
                 runph_line+"\n")

    elif MachineOnWhichToRun == "sharedmemory":
        f.write("#!/bin/bash -l\n")
        f.write("\n")
        f.write ("mpiexec -np 1 pyomo_ns : -np 1 dispatch_srvr : -np "+ \
                 str(workercnt) +" phsolverserver : -np 1 runph -i "+ \
                 instance+" -m "+modelname+" "+runph_line+"\n")

    f.write("\n")
    f.write("## Created by "+str(thisfile)+" at "+str(timestamp)+"\n")
