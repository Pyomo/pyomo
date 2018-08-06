import os
import shutil
import subprocess
import sys
import errno
import stat
import xml.dom.minidom
import re
import glob

if sys.platform.startswith('win'):
    platform = 'win'
elif sys.platform.startswith('linux'):
    platform = 'linux'
elif sys.platform.startswith('darwin'):
    platform = 'darwin'
elif sys.platform.startswith('aix'):
    platform = 'aix'
elif sys.platform.startswith('java'):
    platform = 'java'
elif sys.platform.startswith('cli'):
    platform = 'cli'
else:
    sys.stdout.write( "Unexpected build platform %s\n" % sys.platform )
    sys.exit(1)


def keep(xmlfile, elem_name, attr_name, pattern, dst):
    from xml.etree.ElementTree import ElementTree
    try: 
        rep = re.compile(pattern)
    except TypeError:
        # Create regex pattern if a list is given. 
        # TypeError: unhashable type: 'list'
        rep = re.compile("|".join(pattern))

    tree = ElementTree()
    tree.parse(xmlfile)
    for _node in tree.findall('packages'):
      for node in _node.findall(elem_name):
        if rep.search(node.get(attr_name)):
            _node.remove(node)

    tree.write(dst)
 
#
# Override the default definition of rmtree, to better handle MSWindows errors
# that are associated with read-only files
#
def handleRemoveReadonly(func, path, exc):
  excvalue = exc[1]
  if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
      os.chmod(path, stat.S_IRWXU| stat.S_IRWXG| stat.S_IRWXO) # 0777
      func(path)
  else:
      raise

def rmtree(dir):
    if not os.path.exists(dir):
        return
    if platform == 'win':
        shutil.rmtree(dir, ignore_errors=False, onerror=handleRemoveReadonly)
    else:
        shutil.rmtree(dir)


def perform_build(package, config=None, user='hudson', coverage=False, omit=None, cat='nightly', dest='python', virtualenv=True, virtualenv_args=None):
    perform_install(package, config, user, dest, virtualenv, virtualenv_args)
    if cat is None:
        return
    perform_tests(package, coverage, omit, cat)


def perform_install(package, config=None, user='hudson', dest='python', virtualenv=True, virtualenv_args=None):
    if os.environ.get('WORKSPACE', None) is None:
        sys.stdout.write(
            "\n(INFO) WORKSPACE environment vatiable not found."
            "\n       Assuming WORKSPACE==%s\n\n" % (os.getcwd(),) )
        os.environ['WORKSPACE'] = os.getcwd()

    if os.path.exists(dest):
        if os.path.abspath(sys.executable).startswith(os.path.abspath('python')):
            raise Exception(
                "Python executable used to create the virtual environment:"
                "\n\t    %s\n\tfound within the target installation directory:"
                "\n\t    %s\n\tCowardly refusing to continue installation."
                % ( os.path.abspath(sys.executable), os.path.abspath(dest) ) )
        rmtree(dest)

    # Set the user name for windows builds
    if platform == 'win':
        os.environ['USER'] = user

    if 'CONFIGFILE' in os.environ:
        configfile = os.environ['CONFIGFILE']
    else:
        if config is None:
            config = os.path.join( os.environ['WORKSPACE'],"hudson",package+"-vpy","all.ini" )
        elif os.sep not in config:
            config = os.path.join( os.environ['WORKSPACE'],"hudson",package+"-vpy",config )
        configfile = config

    if 'PYPI_URL' in os.environ:
        if os.environ['PYPI_URL']:
            pypi_url = [ '--pypi-url', os.environ['PYPI_URL'] ]
        else:
            pypi_url = []
    else:
        pypi_url = [ '--pypi-url', 'http://giskard.sandia.gov:8888/pypi',
                     '--trust-pypi-url' ]

    if 'PICO' in os.environ and os.environ['PICO'] == 'yes':
        os.environ['PATH'] = os.pathsep.join(
            os.path.join(os.environ['WORKSPACE'], 'build', 'bin'),
            os.environ['PATH'] )

    # We must cleanup old PYC files for the case that sources were
    # renamed/deleted (coverage will fail if it finds PYC files
    # without corresponding PY files).
    sys.stdout.write("\n")
    for dirpath, dnames, fnames in os.walk(
        os.path.join(os.environ['WORKSPACE'],'src')):
        for fname in fnames:
            if not fname.endswith(('.pyc','.PYC')):
                continue
            if fname[:-1] not in fnames:
                sys.stdout.write("Removing dangling PYC file: %s\n"
                                 % os.path.join(dirpath, fname))
                os.remove(os.path.join(dirpath, fname))

    python=os.environ.get('PYTHON','')
    if python == '':
        python = sys.executable
    elif python[0] == '"':
        python=eval(python)
    sys.stdout.write("\n")
    sys.stdout.write("Installing with Python version %s\n" % sys.version)
    sys.stdout.write("\n")
    # Install
    if virtualenv:
        # Install using vpy_install
        cmd = [
            python,
            os.path.join( os.environ['WORKSPACE'],'vpy','pyutilib','virtualenv', 'vpy_install.py' ),
            '--debug', '-v', '--system-site-packages', '--config', configfile ]
        if pypi_url:
            cmd.extend( pypi_url )
        if virtualenv_args is None:
            cmd.extend(sys.argv[1:])
        else:
            cmd.extend(virtualenv_args)
        cmd.append( os.path.join(os.environ['WORKSPACE'], dest) )
        sys.stdout.write("Running Command: %s\n" % " ".join(cmd))
        sys.stdout.flush()
        if platform == 'win':
            sys.stdout.write( str(subprocess.call(['cmd','/c']+cmd)) + '\n' )
        else:
            sys.stdout.write( str(subprocess.call(cmd)) + '\n' )
    else:
        # Install into a local directory "$WORKSPACE/python"
        sitedir = os.path.join(os.path.abspath(dest),'lib','python'+'.'.join(map(str,sys.version_info[:2])),'site-packages')
        if 'PYTHONPATH' in os.environ:
            os.environ['PYTHONPATH'] = os.environ['PYTHONPATH']+':'+sitedir
        else:
            os.environ['PYTHONPATH'] = sitedir
        os.makedirs(sitedir)
        os.chdir(os.path.join( os.environ['WORKSPACE'],'src' ))
        #print "HERE", os.environ['WORKSPACE']
        for file in glob.glob(os.path.join( os.environ['WORKSPACE'],'src','*')):
            if os.path.isdir(file) and os.path.exists( os.path.join(file,'setup.py') ):
                cmd = [
                    python,
                    'setup.py',
                    'develop', '--no-deps', '--prefix', os.path.join( os.environ['WORKSPACE'],'python') ]
                sys.stdout.write("Running Command: %s\n" % " ".join(cmd))
                sys.stdout.flush()
                os.chdir(file)
                if platform == 'win':
                    sys.stdout.write( str(subprocess.call(['cmd','/c']+cmd)) + '\n' )
                else:
                    sys.stdout.write( str(subprocess.call(cmd)) + '\n' )


def perform_tests(package, coverage=False, omit=None, cat='nightly'):
    if platform == 'win':
        os.environ['PATH'] = os.path.join( os.environ['WORKSPACE'],'python','Scripts' ) + os.pathsep + os.environ['PATH']
    else:
        os.environ['PATH'] = os.path.join( os.environ['WORKSPACE'],'python','bin' ) + os.pathsep + os.environ['PATH']
    if platform == 'win':
        cmd = [os.path.join( os.environ['WORKSPACE'],'python','bin','test.'+package+'.exe' ), '--cat', cat]
    else:
        cmd = [os.path.join( os.environ['WORKSPACE'],'python','bin','test.'+package ), '--cat', cat]
    cmd.append('-v')
    if coverage:
        cmd.extend(['--coverage','--cover-erase'])
    if 'TEST_PACKAGES' in os.environ:
        cmd = cmd + re.split('[ \t]+', os.environ['TEST_PACKAGES'].strip())
    sys.stdout.write( "Running Command: %s\n" % " ".join(cmd) )
    sys.stdout.flush()
    if platform == 'win':
        sys.stdout.write( str(subprocess.call(['cmd','/c']+cmd)) + '\n' )
    else:
        env = os.environ.copy()
        for badness in ['__PYVENV_LAUNCHER__']:
            if badness in env:
                del env[badness]
        #for key_ in sorted(env.keys()):
            #print("%s %s" % (key_, env[key_]))
        sys.stdout.write( str(subprocess.call(cmd, env=env)) + '\n' )
    #
    if not coverage:
        return
    os.chdir(os.path.join( os.environ['WORKSPACE'],'src','pyomo' ))
    libdir = os.path.join( os.environ['WORKSPACE'],'python','lib','*' )
    libdir = libdir + "," + os.path.join( libdir,'site-packages','*' )
    if omit is not None:
        omit = ","+omit
    else:
        omit = ""
    covFName = os.path.join( os.environ['WORKSPACE'],'src','coverage.xml' )
    if platform == 'win':
        cmd = [os.path.join( os.environ['WORKSPACE'],'python','bin','coverage.exe' ), 'xml', '--omit="%s%s"' % (libdir,omit), '-o', covFName ]
    else:
        cmd = [os.path.join( os.environ['WORKSPACE'],'python','bin','coverage' ), 'xml', '--omit="%s%s"' % (libdir,omit), '-o', covFName ]
    sys.stdout.write( "Running Command: %s\n" % " ".join(cmd) )
    sys.stdout.flush()
    if platform == 'win':
        sys.stdout.write( str(subprocess.call(['cmd','/c']+cmd)) + '\n' )
    else:
        #sys.stdout.write( os.getcwd() + '\n')
        #sys.stdout.write( str(subprocess.call(['ls', '-la'])) + '\n')
        sys.stdout.write( str(subprocess.call(cmd)) + '\n' )

    # NB: for Hudson source rendering to work correctly, the XML must include the relative path to the source filenames *from the project workspace*
    try:
        keep(covFName, 'package', 'name', '.*\.tests', covFName)
        doc = xml.dom.minidom.parse(covFName)
    except:
        return
    node = doc.documentElement
    tmp = doc.createElement("sources")
    node.insertBefore(tmp, node.firstChild)
    node.insertBefore(doc.createTextNode("\n\t"), node.firstChild)
    node = tmp
    tmp = doc.createElement("source")
    node.appendChild(tmp)
    tmp.appendChild(doc.createTextNode("src"))
    result = open(covFName, 'w')
    doc.writexml(result)
    result.close()

def build_docs(name):
    for file in glob.glob(os.path.join( os.environ['WORKSPACE'],'src',name+'.doc','GettingStarted','current','PyomoGettingStarted.*' )):
        sys.stdout.write("Removing file %s" % (file,))
        os.remove(file)
    sys.stdout.flush()
    os.chdir(os.path.join( os.environ['WORKSPACE'],'src',name+'.doc','GettingStarted','current' ))
    sys.stdout.write( str(subprocess.call(['make','all'])) + '\n' )
    #
    for file in glob.glob(os.path.join( os.environ['WORKSPACE'],'src',name+'.doc','DevGuide','current','PyomoDevGuide.*' )):
        sys.stdout.write("Removing file %s" % (file,))
        os.remove(file)
    sys.stdout.flush()
    os.chdir(os.path.join( os.environ['WORKSPACE'],'src',name+'.doc','DevGuide','current' ))
    sys.stdout.write( str(subprocess.call(['make','all'])) + '\n' )

if __name__ == '__main__':
    keep('foo.xml', 'package', 'name', '.*\.tests', 'bar.xml')
