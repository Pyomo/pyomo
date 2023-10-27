import os
from zipfile import ZipFile
from pyomo.common import fileutils
from pyomo.common import download
import enum
import math
import csv
from collections.abc import Iterable
import logging


logger = logging.getLogger(__name__)


def get_minlplib_instancedata(target_filename=None):
    """
    Download instancedata.csv from MINLPLib which can be used to get statistics on the problems from minlplib.

    Parameters
    ----------
    target_filename: str
        The full path, including the filename for where to place the downloaded 
        file. The default will be a directory called minlplib in the current 
        working directory and a filename of instancedata.csv.
    """
    if target_filename is None:
        target_filename = os.path.join(os.getcwd(), 'minlplib', 'instancedata.csv')
    download_dir = os.path.dirname(target_filename)

    if os.path.exists(target_filename):
        raise ValueError('A file named {filename} already exists.'.format(filename=target_filename))
    
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    downloader = download.FileDownloader()
    downloader.set_destination_filename(target_filename)
    downloader.get_text_file('http://www.minlplib.org/instancedata.csv')


def _process_acceptable_arg(name, arg, default):
    if arg is None:
        return default
    if isinstance(arg, str):
        if arg not in default:
            raise ValueError("Unrecognized argument for %s: %s" % (name, arg))
        return set([arg])
    if isinstance(arg, Iterable):
        ans = set(str(_) for _ in arg)
        if not ans.issubset(default):
            unknown = default - ans
            raise ValueError("Unrecognized argument for %s: %s" % (name, unknown))
        return ans
    if type(arg) == bool:
        if str(arg) in default:
            return set([str(arg)])
    raise ValueError('unrecognized type for %s: %s' % (name, type(arg)))


def _check_int_arg(arg, _min, _max, arg_name, case_name):
    if arg < _min or arg > _max:
        logger.debug('excluding {case_name} due to {arg_name}'.format(case_name=case_name, arg_name=arg_name))
        return True
    return False


def _check_acceptable(arg, acceptable_set, arg_name, case_name):
    if arg not in acceptable_set:
        logger.debug('excluding {case_name} due to {arg_name}'.format(case_name=case_name, arg_name=arg_name))
        return True
    return False


def filter_minlplib_instances(instancedata_filename=None,
                              min_nvars=0, max_nvars=math.inf,
                              min_nbinvars=0, max_nbinvars=math.inf,
                              min_nintvars=0, max_nintvars=math.inf,
                              min_nnlvars=0, max_nnlvars=math.inf,
                              min_nnlbinvars=0, max_nnlbinvars=math.inf,
                              min_nnlintvars=0, max_nnlintvars=math.inf,
                              min_nobjnz=0, max_nobjnz=math.inf,
                              min_nobjnlnz=0, max_nobjnlnz=math.inf,
                              min_ncons=0, max_ncons=math.inf,
                              min_nlincons=0, max_nlincons=math.inf,
                              min_nquadcons=0, max_nquadcons=math.inf,
                              min_npolynomcons=0, max_npolynomcons=math.inf,
                              min_nsignomcons=0, max_nsignomcons=math.inf,
                              min_ngennlcons=0, max_ngennlcons=math.inf,
                              min_njacobiannz=0, max_njacobiannz=math.inf,
                              min_njacobiannlnz=0, max_njacobiannlnz=math.inf,
                              min_nlaghessiannz=0, max_nlaghessiannz=math.inf,
                              min_nlaghessiandiagnz=0, max_nlaghessiandiagnz=math.inf,
                              min_nsemi=0, max_nsemi=math.inf,
                              min_nnlsemi=0, max_nnlsemi=math.inf,
                              min_nsos1=0, max_nsos1=math.inf,
                              min_nsos2=0, max_nsos2=math.inf,
                              acceptable_formats=None,
                              acceptable_probtype=None,
                              acceptable_objtype=None,
                              acceptable_objcurvature=None,
                              acceptable_conscurvature=None,
                              acceptable_convex=None):
    """
    This function filters problems from MINLPLib based on
    instancedata.csv from MINLPLib and the conditions specified
    through the function arguments. The function argument names
    correspond to column headings from instancedata.csv. The 
    arguments starting with min or max require int or float inputs. 
    The arguments starting with acceptable require either a 
    string or an iterable of strings. See the MINLPLib documentation 
    for acceptable values.
    """
    if instancedata_filename is None:
        instancedata_filename = os.path.join(os.getcwd(), 'minlplib', 'instancedata.csv')

    if not os.path.exists(instancedata_filename):
        raise RuntimeError('{filename} does not exist. Please use get_minlplib_instancedata() first or specify the location of the MINLPLib instancedata.csv with the instancedata_filename argument.'.format(filename=instancedata_filename))

    acceptable_formats = _process_acceptable_arg(
        'acceptable_formats',
        acceptable_formats,
        set(['ams', 'gms', 'lp', 'mod', 'nl', 'osil', 'pip'])
    )

    default_acceptable_probtype = set()
    for pre in ['B', 'I', 'MI', 'MB', 'S', '']:
        for post in ['NLP', 'QCQP', 'QP', 'QCP', 'P']:
            default_acceptable_probtype.add(pre + post)
    acceptable_probtype = _process_acceptable_arg(
        'acceptable_probtype',
        acceptable_probtype,
        default_acceptable_probtype
        )

    acceptable_objtype = _process_acceptable_arg(
        'acceptable_objtype',
        acceptable_objtype,
        set(['constant', 'linear', 'quadratic', 'polynomial', 'signomial', 'nonlinear'])
        )

    acceptable_objcurvature = _process_acceptable_arg(
        'acceptable_objcurvature',
        acceptable_objcurvature,
        set(['linear', 'convex', 'concave', 'indefinite', 'nonconvex', 'nonconcave', 'unknown'])
        )

    acceptable_conscurvature = _process_acceptable_arg(
        'acceptable_conscurvature',
        acceptable_conscurvature,
        set(['linear', 'convex', 'concave', 'indefinite', 'nonconvex', 'nonconcave', 'unknown'])
        )

    acceptable_convex = _process_acceptable_arg(
        'acceptable_convex',
        acceptable_convex,
        set(['True', 'False', ''])
        )

    int_arg_name_list = [
        'nvars',
        'nbinvars',
        'nintvars',
        'nnlvars',
        'nnlbinvars',
        'nnlintvars',
        'nobjnz',
        'nobjnlnz',
        'ncons',
        'nlincons',
        'nquadcons',
        'npolynomcons',
        'nsignomcons',
        'ngennlcons',
        'njacobiannz',
        'njacobiannlnz',
        'nlaghessiannz',
        'nlaghessiandiagnz',
        'nsemi',
        'nnlsemi',
        'nsos1',
        'nsos2',
    ]
    min_list = [
        min_nvars,
        min_nbinvars,
        min_nintvars,
        min_nnlvars,
        min_nnlbinvars,
        min_nnlintvars,
        min_nobjnz,
        min_nobjnlnz,
        min_ncons,
        min_nlincons,
        min_nquadcons,
        min_npolynomcons,
        min_nsignomcons,
        min_ngennlcons,
        min_njacobiannz,
        min_njacobiannlnz,
        min_nlaghessiannz,
        min_nlaghessiandiagnz,
        min_nsemi,
        min_nnlsemi,
        min_nsos1,
        min_nsos2,
    ]
    max_list = [
        max_nvars,
        max_nbinvars,
        max_nintvars,
        max_nnlvars,
        max_nnlbinvars,
        max_nnlintvars,
        max_nobjnz,
        max_nobjnlnz,
        max_ncons,
        max_nlincons,
        max_nquadcons,
        max_npolynomcons,
        max_nsignomcons,
        max_ngennlcons,
        max_njacobiannz,
        max_njacobiannlnz,
        max_nlaghessiannz,
        max_nlaghessiandiagnz,
        max_nsemi,
        max_nnlsemi,
        max_nsos1,
        max_nsos2,
    ]

    acceptable_arg_name_list = [
        'probtype',
        'objtype',
        'objcurvature',
        'conscurvature',
        'convex'
    ]
    acceptable_set_list = [
        acceptable_probtype,
        acceptable_objtype,
        acceptable_objcurvature,
        acceptable_conscurvature,
        acceptable_convex
    ]

    with open(instancedata_filename, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=';')
        headings = {column: ndx for ndx, column in enumerate(next(reader))}
        rows = [row for row in reader]

    cases = list()
    for ndx, row in enumerate(rows):
        if len(row) == 0:
            continue
        
        case_name = row[headings['name']]

        available_formats = row[headings['formats']]
        available_formats = available_formats.replace('set([', '')
        available_formats = available_formats.replace('])', '')
        available_formats = available_formats.replace('{', '')
        available_formats = available_formats.replace('}', '')
        available_formats = available_formats.replace(' ', '')
        available_formats = available_formats.replace("'", '')
        available_formats = available_formats.split(',')
        available_formats = set(available_formats)

        should_continue = False

        if len(acceptable_formats.intersection(available_formats)) == 0:
            logger.debug('excluding {case} due to available_formats'.format(case=case_name))
            should_continue = True

        for ndx, acceptable_arg_name in enumerate(acceptable_arg_name_list):
            acceptable_set = acceptable_set_list[ndx]
            arg = row[headings[acceptable_arg_name]]
            if _check_acceptable(arg=arg,
                                 acceptable_set=acceptable_set,
                                 arg_name=acceptable_arg_name,
                                 case_name=case_name):
                should_continue = True

        for ndx, arg_name in enumerate(int_arg_name_list):
            _min = min_list[ndx]
            _max = max_list[ndx]
            arg = int(row[headings[arg_name]])
            if _check_int_arg(arg=arg,
                              _min=_min,
                              _max=_max,
                              arg_name=arg_name,
                              case_name=case_name):
                should_continue = True

        if should_continue:
            continue

        cases.append(case_name)

    return cases


def get_minlplib(download_dir=None, format='osil', problem_name=None):
    """
    Download MINLPLib

    Parameters
    ----------
    download_dir: str
        The directory in which to place the downloaded files. The default will be a 
        current_working_directory/minlplib/file_format/.
    format: str
        The file format requested. Options are ams, gms, lp, mod, nl, osil, and pip
    problem_name: None or str
        If problem_name is None, then the entire zip file will be downloaded 
        and extracted (all problems with the specified format). If a single problem 
        needs to be downloaded, then the name of the problem can be specified. 
        This can be significantly faster than downloading all of the problems. 
        However, individual problems are not compressed, so downloading multiple 
        individual problems can quickly become expensive.
    """
    if download_dir is None:
        download_dir = os.path.join(os.getcwd(), 'minlplib', format)

    if problem_name is None:
        if os.path.exists(download_dir):
            raise ValueError(
                'The specified download_dir already exists: ' + download_dir)

        os.makedirs(download_dir)
        downloader = download.FileDownloader()
        zip_dirname = os.path.join(download_dir, 'minlplib_'+format)
        downloader.set_destination_filename(zip_dirname)
        downloader.get_zip_archive('http://www.minlplib.org/minlplib_'+format+'.zip')
        for i in os.listdir(os.path.join(download_dir, 'minlplib_'+format, 'minlplib', format)):
            os.rename(os.path.join(download_dir, 'minlplib_'+format, 'minlplib', format, i), os.path.join(download_dir, i))
        os.rmdir(os.path.join(download_dir, 'minlplib_'+format, 'minlplib', format))
        os.rmdir(os.path.join(download_dir, 'minlplib_'+format, 'minlplib'))
        os.rmdir(os.path.join(download_dir, 'minlplib_'+format))
    else:
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        target_filename = os.path.join(download_dir, problem_name + '.' + format)
        if os.path.exists(target_filename):
            raise ValueError(f'The target filename ({target_filename}) already exists')
        downloader = download.FileDownloader()
        downloader.set_destination_filename(target_filename)
        downloader.get_binary_file('http://www.minlplib.org/'+format+'/'+problem_name+'.'+format)
        
