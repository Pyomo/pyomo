#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import sys
import re
import copy
import logging

from pyomo.common.log import is_debug_set
from pyomo.common.collections import Options, OrderedDict
from pyomo.common.errors import ApplicationError
from pyutilib.misc import flatten

from pyomo.dataportal.parse_datacmds import (
    parse_data_commands, _re_number
)
from pyomo.dataportal.factory import DataManagerFactory, UnknownDataManager
from pyomo.core.base.set import UnknownSetDimen

from six.moves import xrange
try:
    unicode
except:
    unicode = str
try:
    long
    numlist = {bool, int, float, long}
except:
    numlist = {bool, int, float}

logger = logging.getLogger('pyomo.core')

global Lineno
global Filename

_num_pattern = re.compile("^("+_re_number+")$")
_str_false_values = {'False','false','FALSE'}
_str_bool_values = {'True','true','TRUE'}
_str_bool_values.update(_str_false_values)

def _guess_set_dimen(index):
    d = 0
    # Look through the subsets of this index and get their dimen
    for subset in index.subsets():
        sub_d = subset.dimen
        # If the subset has an unknown dimen, then look at the subset's
        # domain to guess the dimen.
        if sub_d is UnknownSetDimen:
            for domain_subset in subset.domain.subsets():
                sub_d = domain_subset.domain.dimen
                if sub_d in (UnknownSetDimen, None):
                    # We will guess that None / Unknown domains are dimen==1
                    d += 1
                else:
                    d += sub_d
        elif sub_d is None:
            return None
        else:
            d += sub_d
    return d

def _process_token(token):
    #print("TOKEN:", token, type(token))
    if type(token) is tuple:
        return tuple(_process_token(i) for i in token)
    elif type(token) in numlist:
        return token
    elif token in _str_bool_values:
        return token not in _str_false_values
    elif token[0] == '"' and token[-1] == '"':
        # Strip "flag" quotation characters
        return token[1:-1]
    elif token[0] == '[' and token[-1] == ']':
        vals = []
        token = token[1:-1]
        for item in token.split(","):
            if item[0] in '"\'' and item[0] == item[-1]:
                vals.append( item[1:-1] )
            elif _num_pattern.match(item):
                _num = float(item)
                if '.' in item:
                    vals.append(_num)
                else:
                    _int = int(_num)
                    vals.append(_int if _int == _num else _num)
            else:
                vals.append( item )
        return tuple(vals)
    elif _num_pattern.match(token):
        _num = float(token)
        if '.' in token:
            return _num
        else:
            _int = int(_num)
            return _int if _int == _num else _num
    else:
        return token


def _preprocess_data(cmd):
    """
    Called by _process_data() to (1) combine tokens that comprise a tuple
    and (2) combine the ':' token with the previous token
    """
    generate_debug_messages = is_debug_set(logger)
    if generate_debug_messages:
        logger.debug("_preprocess_data(start) %s",cmd)
    state = 0
    newcmd=[]
    tpl = []
    for token in cmd:
        if state == 0:
            if type(token) in numlist:
                newcmd.append(token)
            elif token == ',':
                raise ValueError("Unexpected comma outside of (), {} or [] declarations")
            elif token == '(':
                state = 1
            elif token == ')':
                raise ValueError("Unexpected ')' that does not follow a '('")
            elif token == '{':
                state = 2
            elif token == '}':
                raise ValueError("Unexpected '}' that does not follow a '{'")
            elif token == '[':
                state = 3
            elif token == ']':
                raise ValueError("Unexpected ']' that does not follow a '['")
            else:
                newcmd.append(_process_token(token))

        elif state == 1:
            # After a '('
            if type(token) in numlist:
                tpl.append(token)
            elif token == ',':
                pass
            elif token == '(':
                raise ValueError("Two '('s follow each other in the data")
            elif token == ')':
                newcmd.append( tuple(tpl) )
                tpl = []
                state = 0
            else:
                tpl.append(_process_token(token))

        elif state == 2:
            # After a '{'
            if type(token) in numlist:
                tpl.append(token)
            elif token == ',':
                pass
            elif token == '{':
                raise ValueError("Two '{'s follow each other in the data")
            elif token == '}':
                newcmd.append( tpl )    # Keep this as a list, so we can distinguish it while parsing tables
                tpl = []
                state = 0
            else:
                tpl.append(_process_token(token))

        elif state == 3:
            # After a '['
            if type(token) in numlist:
                tpl.append(token)
            elif token == ',':
                pass
            elif token == '[':
                raise ValueError("Two '['s follow each other in the data")
            elif token == ']':
                newcmd.append( tuple(tpl) )
                tpl = []
                state = 0
            else:
                tpl.append(_process_token(token))

    if state == 1:
        raise ValueError("Data ends without tuple ending")
    elif state == 2:
        raise ValueError("Data ends without braces ending")
    elif state == 3:
        raise ValueError("Data ends without bracket ending")
    if generate_debug_messages:
        logger.debug("_preprocess_data(end) %s", newcmd)
    return newcmd


def _process_set(cmd, _model, _data):
    """
    Called by _process_data() to process a set declaration.
    """
    #print("SET %s" % cmd)
    generate_debug_messages = is_debug_set(logger)
    if generate_debug_messages:
        logger.debug("DEBUG: _process_set(start) %s",cmd)
    #
    # Process a set
    #
    if type(cmd[2]) is tuple:
        #
        # An indexed set
        #
        ndx=cmd[2]
        if len(ndx) == 0:
            # At this point, if the index is an empty tuple, then there is an
            # issue with the specification of this indexed set.
            raise ValueError("Illegal indexed set specification encountered: "+str(cmd[1]))
        elif len(ndx) == 1:
            ndx=ndx[0]
        if cmd[1] not in _data:
            _data[cmd[1]] = {}
        _data[cmd[1]][ndx] = _process_set_data(cmd[4:], cmd[1], _model)

    elif cmd[2] == ":":
        #
        # A tabular set
        #
        _data[cmd[1]] = {}
        _data[cmd[1]][None] = []
        i=3
        while cmd[i] != ":=":
            i += 1
        ndx1 = cmd[3:i]
        i += 1
        while i<len(cmd):
            ndx=cmd[i]
            for j in xrange(0,len(ndx1)):
                if cmd[i+j+1] == "+":
                    #print("DATA %s %s" % (ndx1[j], cmd[i]))
                    _data[cmd[1]][None].append((ndx1[j], cmd[i]))
            i += len(ndx1)+1
    else:
        #
        # Processing a general set
        #
        _data[cmd[1]] = {}
        _data[cmd[1]][None] = _process_set_data(cmd[3:], cmd[1], _model)


def _process_set_data(cmd, sname, _model):
    """
    Called by _process_set() to process set data.
    """
    generate_debug_messages = is_debug_set(logger)
    if generate_debug_messages:
        logger.debug("DEBUG: _process_set_data(start) %s",cmd)
    if len(cmd) == 0:
        return []
    ans=[]
    i=0
    template=None
    ndx=[]
    template = []
    while i<len(cmd):
        if type(cmd[i]) is not tuple:
            if len(ndx) == 0:
                ans.append(cmd[i])
            else:
                #
                # Use the ndx to create a tuple.  This list
                # contains the indices of the values that need
                # to be filled-in
                #
                tmpval=template
                for kk in range(len(ndx)):
                    if i == len(cmd):
                        raise IOError("Expected another set value to flush out a tuple pattern!")
                    tmpval[ndx[kk]] = cmd[i]
                    i += 1
                ans.append(tuple(tmpval))
                continue
        elif "*" not in cmd[i]:
            ans.append(cmd[i])
        else:
            template=list(cmd[i])
            ndx=[]
            for kk in range(len(template)):
                if template[kk] == '*':
                    ndx.append(kk)
        i += 1
    if generate_debug_messages:
        logger.debug("DEBUG: _process_set_data(end) %s",ans)
    return ans


def _process_param(cmd, _model, _data, _default, index=None, param=None, ncolumns=None):
    """
    Called by _process_data to process data for a Parameter declaration
    """
    #print('PARAM %s index=%s ncolumns=%s' %(cmd, index, ncolumns))
    generate_debug_messages = is_debug_set(logger)
    if generate_debug_messages:
        logger.debug("DEBUG: _process_param(start) %s",cmd)
    #
    # Process parameters
    #
    dflt = None
    singledef = True
    cmd = cmd[1:]
    if cmd[0] == ":":
        singledef = False
        cmd = cmd[1:]
    #print "SINGLEDEF", singledef
    if singledef:
        pname = cmd[0]
        cmd = cmd[1:]
        if len(cmd) >= 2 and cmd[0] == "default":
            dflt = cmd[1]
            cmd = cmd[2:]
        if dflt != None:
            _default[pname] = dflt
        if cmd[0] == ":=":
            cmd = cmd[1:]
        transpose = False
        if cmd[0] == "(tr)":
            transpose = True
            if cmd[1] == ":":
                cmd = cmd[1:]
            else:
                cmd[0] = ":"
        if cmd[0] != ":":
            #print "HERE YYY", pname, transpose, _model, ncolumns
            if generate_debug_messages:
                logger.debug("DEBUG: _process_param (singledef without :...:=) %s",cmd)
            cmd = _apply_templates(cmd)
            #print 'cmd',cmd
            if not transpose:
                if pname not in _data:
                    _data[pname] = {}
                if not ncolumns is None:
                    finaldata = _process_data_list(pname, ncolumns-1, cmd)
                elif not _model is None:
                    _param = getattr(_model, pname)
                    _dim = _param.dim()
                    if _dim is UnknownSetDimen:
                        _dim = _guess_set_dimen(_param.index_set())
                    finaldata = _process_data_list(pname, _dim, cmd)
                else:
                    finaldata = _process_data_list(pname, 1, cmd)
                for key in finaldata:
                    _data[pname][key]=finaldata[key]
            else:
                tmp = ["param", pname, ":="]
                i=1
                while i < len(cmd):
                    i0 = i
                    while cmd[i] != ":=":
                        i=i+1
                    ncol = i - i0 + 1
                    lcmd = i
                    while lcmd < len(cmd) and cmd[lcmd] != ":":
                        lcmd += 1
                    j0 = i0 - 1
                    for j in range(1,ncol):
                        ii = 1 + i
                        kk = ii + j
                        while kk < lcmd:
                            if cmd[kk] != ".":
                            #if 1>0:
                                tmp.append(copy.copy(cmd[j+j0]))
                                tmp.append(copy.copy(cmd[ii]))
                                tmp.append(copy.copy(cmd[kk]))
                            ii = ii + ncol
                            kk = kk + ncol
                    i = lcmd + 1
                _process_param(tmp, _model, _data, _default, index=index, param=param, ncolumns=ncolumns)
        else:
            tmp = ["param", pname, ":="]
            if param is None:
                param = [ pname ]
            i=1
            if generate_debug_messages:
                logger.debug("DEBUG: _process_param (singledef with :...:=) %s",cmd)
            while i < len(cmd):
                i0 = i
                while i<len(cmd) and cmd[i] != ":=":
                    i=i+1
                if i==len(cmd):
                    raise ValueError("ERROR: Trouble on line "+str(Lineno)+" of file "+Filename)
                ncol = i - i0 + 1
                lcmd = i
                while lcmd < len(cmd) and cmd[lcmd] != ":":
                    lcmd += 1
                j0 = i0 - 1
                for j in range(1,ncol):
                    ii = 1 + i
                    kk = ii + j
                    while kk < lcmd:
                        if cmd[kk] != ".":
                            if transpose:
                                tmp.append(copy.copy(cmd[j+j0]))
                                tmp.append(copy.copy(cmd[ii]))
                            else:
                                tmp.append(copy.copy(cmd[ii]))
                                tmp.append(copy.copy(cmd[j+j0]))
                            tmp.append(copy.copy(cmd[kk]))
                        ii = ii + ncol
                        kk = kk + ncol
                i = lcmd + 1
                _process_param(tmp, _model, _data, _default, index=index, param=param[0], ncolumns=3)

    else:
        if generate_debug_messages:
            logger.debug("DEBUG: _process_param (cmd[0]=='param:') %s",cmd)
        i=0
        nsets=0
        while i<len(cmd):
            if cmd[i] == ':=':
                i = -1
                break
            if cmd[i] == ":":
                nsets = i
                break
            i += 1
        nparams=0
        _i = i+1
        while i<len(cmd):
            if cmd[i] == ':=':
                nparams = i-_i
                break
            i += 1
        if i==len(cmd):
            raise ValueError("Trouble on data file line "+str(Lineno)+" of file "+Filename)
        if generate_debug_messages:
            logger.debug("NSets %d",nsets)
        Lcmd = len(cmd)
        j=0
        d = 1
        #print "HERE", nsets, nparams
        #
        # Process sets first
        #
        while j<nsets:
            # NOTE: I'm pretty sure that nsets is always equal to 1
            sname = cmd[j]
            if not ncolumns is None:
                d = ncolumns-nparams
            elif _model is None:
                d = 1
            else:
                index = getattr(_model, sname)
                d = _guess_set_dimen(index)
            #print "SET",sname,d,_model#,getattr(_model,sname).dimen, type(index)
            #d = getattr(_model,sname).dimen
            np = i-1
            if generate_debug_messages:
                logger.debug("I %d, J %d, SName %s, d %d",i,j,sname,d)
            dnp = d + np - 1
            #k0 = i + d - 2
            ii = i + j + 1
            tmp = [ "set", cmd[j], ":=" ]
            while ii < Lcmd:
                if d > 1:
                    _tmp = []
                    for dd in range(0,d):
                        _tmp.append(copy.copy(cmd[ii+dd]))
                    tmp.append(tuple(_tmp))
                else:
                    for dd in range(0,d):
                        tmp.append(copy.copy(cmd[ii+dd]))
                ii += dnp
            _process_set(tmp, _model, _data)
            j += 1
        if nsets > 0:
            j += 1
        #
        # Process parameters second
        #
        #print "HERE", cmd
        #print "HERE", param
        #print "JI",j,i # XXX
        jstart = j
        if param is None:
            param = []
            _j = j
            while _j < i:
                param.append( cmd[_j] )
                _j += 1
        while j < i:
            #print "HERE", i, j, jstart, cmd[j]
            pname = param[j-jstart]
            if generate_debug_messages:
                logger.debug("I %d, J %d, Pname %s",i,j,pname)
            if not ncolumns is None:
                d = ncolumns - nparams
            elif _model is None:
                d = 1
            else:
                _param = getattr(_model, pname)
                d = _param.dim()
                if d is UnknownSetDimen:
                    d = _guess_set_dimen(_param.index_set())
            if nsets > 0:
                np = i-1
                dnp = d+np-1
                ii = i + 1
                kk = i + d + j-1
            else:
                np = i
                dnp = d + np
                ii = i + 1
                kk = np + 1 + d + nsets + j
            #print cmd[ii], d, np, dnp, ii, kk
            tmp = [ "param", pname, ":=" ]
            if generate_debug_messages:
                logger.debug('dnp %d\nnp %d', dnp, np)
            while kk < Lcmd:
                if generate_debug_messages:
                    logger.debug("kk %d, ii %d",kk,ii)
                iid = ii + d
                while ii < iid:
                    tmp.append(copy.copy(cmd[ii]))
                    ii += 1
                ii += dnp-d
                tmp.append(copy.copy(cmd[kk]))
                kk += dnp
            #print "TMP", tmp, ncolumns-nparams+1
            if not ncolumns is None:
                nc = ncolumns-nparams+1
            else:
                nc = None
            _process_param(tmp, _model, _data, _default, index=index, param=param[j-jstart], ncolumns=nc)
            j += 1


def _apply_templates(cmd):
    template = []
    ilist = set()
    ans = []
    i = 0
    while i < len(cmd):
    #print i, len(cmd), cmd[i], ilist, template, ans
        if type(cmd[i]) is tuple and '*' in cmd[i]:
            j = i
            tmp=list(cmd[j])
            nindex = len(tmp)
            template=tmp
            ilist = set()
            for kk in range(nindex):
                if tmp[kk] == '*':
                    ilist.add(kk)
        elif len(ilist) == 0:
            ans.append(cmd[i])
        else:
            for kk in range(len(template)):
                if kk in ilist:
                    ans.append(cmd[i])
                    i += 1
                else:
                    ans.append(template[kk])
            ans.append(cmd[i])
        i += 1
    return ans


def _process_data_list(param_name, dim, cmd):
    """\
 Called by _process_param() to process a list of data for a Parameter.
 """
    generate_debug_messages = is_debug_set(logger)
    if generate_debug_messages:
        logger.debug("process_data_list %d %s",dim,cmd)

    if len(cmd) % (dim+1) != 0:
        msg = "Parameter '%s' defined with '%d' dimensions, " \
              "but data has '%d' values: %s."
        msg = msg % (param_name, dim, len(cmd), cmd)
        if len(cmd) % (dim+1) == dim:
            msg += " Are you missing a value for a %d-dimensional index?" % dim
        elif len(cmd) % dim == 0:
            msg += " Are you missing the values for %d-dimensional indices?" % dim
        else:
            msg += " Data declaration must be given in multiples of %d." % (dim + 1)
        raise ValueError(msg)

    ans={}
    if dim == 0:
        ans[None]=cmd[0]
        return ans
    i=0
    while i < len(cmd):
        if dim > 1:
            ndx = tuple(cmd[i:i+dim])
        else:
            ndx = cmd[i]
        if cmd[i+dim] != ".":
            ans[ndx] = cmd[i+dim]
        i += dim+1
    return ans


def _process_include(cmd, _model, _data, _default, options=None):
    if len(cmd) == 1:
        raise IOError("Cannot execute 'include' command without a filename")
    if len(cmd) > 2:
        raise IOError("The 'include' command only accepts a single filename")

    global Filename
    Filename = cmd[1]
    global Lineno
    Lineno = 0
    try:
        scenarios = parse_data_commands(filename=cmd[1])
    except IOError:
        raise
        err = sys.exc_info()[1]
        raise IOError("Error parsing file '%s': %s" % (Filename, str(err)))
    if scenarios is None:
        return False
    for scenario in scenarios:
        for cmd in scenarios[scenario]:
            if scenario not in _data:
                _data[scenario] = {}
            if cmd[0] in ('include', 'load'):
                _tmpdata = {}
                _process_data(cmd, _model, _tmpdata, _default, Filename, Lineno)
                if scenario is None:
                    for key in _tmpdata:
                        if key in _data:
                            _data[key].update(_tmpdata[key])
                        else:
                            _data[key] = _tmpdata[key]
                else:
                    for key in _tmpdata:
                        if key is None:
                            _data[scenario].update(_tmpdata[key])
                        else:
                            raise IOError("Cannot define a scenario within another scenario")
            else:
                _process_data(cmd, _model, _data[scenario], _default, Filename, Lineno)
    return True


def _process_table(cmd, _model, _data, _default, options=None):
    #print("TABLE %s" % cmd)
    #
    _options = {}
    _set = OrderedDict()
    _param = OrderedDict()
    _labels = []

    _cmd = cmd[1]
    _cmd_len = len(_cmd)
    name = None
    i = 0
    while i < _cmd_len:
        try:
            #print("CMD i=%s cmd=%s" % (i, _cmd[i:]))
            #
            # This should not be error prone, so we treat errors
            # with a general exception
            #

            #
            # Processing labels
            #
            if _cmd[i] == ':':
                i += 1
                while i < _cmd_len:
                    _labels.append(_cmd[i])
                    i += 1
                continue
            #
            # Processing options
            #
            name = _cmd[i]
            if i+1 == _cmd_len:
                _param[name] = []
                _labels = ['Z']
                i += 1
                continue
            if _cmd[i+1] == '=':
                if type(_cmd[i+2]) is list:
                    _set[name] = _cmd[i+2]
                else:
                    _options[name] = _cmd[i+2]
                i += 3
                continue
            # This should be a parameter declaration
            if not type(_cmd[i+1]) is tuple:
                raise IOError
            if i+2 < _cmd_len and _cmd[i+2] == '=':
                _param[name] = (_cmd[i+1], _cmd[i+3][0])
                i += 4
            else:
                _param[name] = _cmd[i+1]
                i += 2
        except:
            raise IOError("Error parsing table options: %s" % name)


    #print("_options %s" % _options)
    #print("_set %s" % _set)
    #print("_param %s" % _param)
    #print("_labels %s" % _labels)
#
    options = Options(**_options)
    for key in options:
        if not key in ['columns']:
            raise ValueError("Unknown table option '%s'" % key)
    #
    ncolumns=options.columns
    if ncolumns is None:
        ncolumns = len(_labels)
        if ncolumns == 0:
            if not (len(_set) == 1 and len(_set[_set.keys()[0]]) == 0):
                raise IOError("Must specify either the 'columns' option or column headers")
            else:
                ncolumns=1
    else:
        ncolumns = int(ncolumns)
    #
    data = cmd[2]
    Ldata = len(cmd[2])
    #
    cmap = {}
    if len(_labels) == 0:
        for i in range(ncolumns):
            cmap[i+1] = i
        for label in _param:
            ndx = cmap[_param[label][1]]
            if ndx < 0 or ndx >= ncolumns:
                raise IOError("Bad column value %s for data %s" % (str(ndx), label))
            cmap[label] = ndx
            _param[label] = _param[label][0]
    else:
        i = 0
        for label in _labels:
            cmap[label] = i
            i += 1
    #print("CMAP %s" % cmap)
    #
    #print("_param %s" % _param)
    #print("_set %s" % _set)
    for sname in _set:
        # Creating set sname
        cols = _set[sname]
        tmp = []
        for col in cols:
            if not col in cmap:
                raise IOError("Unexpected table column '%s' for index set '%s'" % (col, sname))
            tmp.append(cmap[col])
        if not sname in cmap:
            cmap[sname] = tmp
        cols = flatten(tmp)
        #
        _cmd = ['set', sname, ':=']
        i = 0
        while i < Ldata:
            row = []
            #print("COLS %s  NCOLS %d" % (cols, ncolumns))
            for col in cols:
                #print("Y %s %s" % (i, col))
                row.append( data[i+col] )
            if len(row) > 1:
                    _cmd.append( tuple(row) )
            else:
                    _cmd.append( row[0] )
            i += ncolumns
        #print("_data %s" % _data)
        _process_set(_cmd, _model, _data)
    #
    #print("CMAP %s" % cmap)
    _i=0
    if ncolumns == 0:
        raise IOError
    for vname in _param:
        _i += 1
        # create value vname
        cols = _param[vname]
        tmp = []
        for col in cols:
            #print("COL %s" % col)
            if not col in cmap:
                raise IOError("Unexpected table column '%s' for table value '%s'" % (col, vname))
            tmp.append(cmap[col])
        #print("X %s %s" % (len(cols), tmp))
        cols = flatten(tmp)
        #print("X %s" % len(cols))
        #print("VNAME %s %s" % (vname, cmap[vname]))
        if vname in cmap:
            cols.append(cmap[vname])
        else:
            cols.append( ncolumns-1 - (len(_param)-_i) )
        #print("X %s" % len(cols))
        #
        _cmd = ['param', vname, ':=']
        i = 0
        while i < Ldata:
            #print("HERE %s %s %s" % (i, cols, ncolumns))
            for col in cols:
                _cmd.append( data[i+col] )
            i += ncolumns
        #print("HERE %s" % _cmd)
        #print("_data %s" % _data)
        _process_param(_cmd, _model, _data, None, ncolumns=len(cols))


def _process_load(cmd, _model, _data, _default, options=None):
    #print("LOAD %s" % cmd)
    from pyomo.core import Set

    _cmd_len = len(cmd)
    _options = {}
    _options['filename'] = cmd[1]
    i=2
    while cmd[i] != ':':
        _options[cmd[i]] = cmd[i+2]
        i += 3
    i += 1
    _Index = (None, [])
    if type(cmd[i]) is tuple:
        _Index = (None, cmd[i])
        i += 1
    elif i+1 < _cmd_len and cmd[i+1] == '=':
        _Index = (cmd[i], cmd[i+2])
        i += 3
    _smap = OrderedDict()
    while i<_cmd_len:
        if i+2 < _cmd_len and cmd[i+1] == '=':
            _smap[cmd[i+2]] = cmd[i]
            i += 3
        else:
            _smap[cmd[i]] = cmd[i]
            i += 1

    if len(cmd) < 2:
        raise IOError("The 'load' command must specify a filename")

    options = Options(**_options)
    for key in options:
        if not key in ['range','filename','format','using','driver','query','table','user','password','database']:
            raise ValueError("Unknown load option '%s'" % key)

    global Filename
    Filename = options.filename

    global Lineno
    Lineno = 0
    #
    # TODO: process mapping info
    #
    if options.using is None:
        tmp = options.filename.split(".")[-1]
        data = DataManagerFactory(tmp)
        if (data is None) or \
           isinstance(data, UnknownDataManager):
            raise ApplicationError("Data manager '%s' is not available." % tmp)
    else:
        try:
            data = DataManagerFactory(options.using)
        except:
            data = None
        if (data is None) or \
           isinstance(data, UnknownDataManager):
            raise ApplicationError("Data manager '%s' is not available." % options.using)
    set_name=None
    #
    # Create symbol map
    #
    symb_map = _smap
    if len(symb_map) == 0:
        raise IOError("Must specify at least one set or parameter name that will be loaded")
    #
    # Process index data
    #
    _index=None
    index_name=_Index[0]
    _select = None
    #
    # Set the 'set name' based on the format
    #
    _set = None
    if options.format == 'set' or options.format == 'set_array':
        if len(_smap) != 1:
            raise IOError("A single set name must be specified when using format '%s'" % options.format)
        set_name=list(_smap.keys())[0]
        _set = set_name
    #
    # Set the 'param name' based on the format
    #
    _param = None
    if options.format == 'transposed_array' or options.format == 'array' or options.format == 'param':
        if len(_smap) != 1:
            raise IOError("A single parameter name must be specified when using format '%s'" % options.format)
    if options.format in ('transposed_array', 'array', 'param', None):
        if _Index[0] is None:
            _index = None
        else:
            _index = _Index[0]
        _param = []
        _select = list(_Index[1])
        for key in _smap:
            _param.append( _smap[key] )
            _select.append( key )
    if options.format in ('transposed_array', 'array'):
        _select = None

    #print "YYY", _param, options
    if not _param is None and len(_param) == 1 and not _model is None and isinstance(getattr(_model, _param[0]), Set):
        _select = None
        _set = _param[0]
        _param = None
        _index = None

    #print "SELECT", _param, _select
    #
    data.initialize(model=options.model, filename=options.filename, index=_index, index_name=index_name, param_name=symb_map, set=_set, param=_param, format=options.format, range=options.range, query=options.query, using=options.using, table=options.table, select=_select,user=options.user,password=options.password,database=options.database)
    #
    data.open()
    try:
        data.read()
    except Exception:
        data.close()
        raise
    data.close()
    data.process(_model, _data, _default)


def _process_data(cmd, _model, _data, _default, Filename_, Lineno_=0, index=None, set=None, param=None, ncolumns=None):
    """
    Called by import_file() to (1) preprocess data and (2) call
    subroutines to process different types of data
    """
    #print("CMD %s" %cmd)
    global Lineno
    global Filename
    Lineno=Lineno_
    Filename=Filename_
    generate_debug_messages = is_debug_set(logger)
    if generate_debug_messages:
        logger.debug("DEBUG: _process_data (start) %s",cmd)
    if len(cmd) == 0:                       #pragma:nocover
        raise ValueError("ERROR: Empty list passed to Model::_process_data")

    if cmd[0] == "data":
        return True

    if cmd[0] == "end":
        return False

    if cmd[0].startswith('set'):
        cmd = _preprocess_data(cmd)
        _process_set(cmd, _model, _data)

    elif cmd[0].startswith('param'):
        cmd = _preprocess_data(cmd)
        _process_param(cmd, _model, _data, _default, index=index, param=param, ncolumns=ncolumns)

    elif cmd[0] == 'include':
        cmd = _preprocess_data(cmd)
        _process_include(cmd, _model, _data, _default)

    elif cmd[0] == 'load':
        cmd = _preprocess_data(cmd)
        _process_load(cmd, _model, _data, _default)

    elif cmd[0] == 'table':
        cmd = [cmd[0], _preprocess_data(cmd[1]), _preprocess_data(cmd[2])]
        _process_table(cmd, _model, _data, _default)

    else:
        raise IOError("ERROR: Unknown data command: "+" ".join(cmd))

    return True
