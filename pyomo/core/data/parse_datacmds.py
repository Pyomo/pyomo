#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ['parse_data_commands']

import sys
import os
import os.path
import ply.lex as lex
import ply.yacc as yacc
from inspect import getfile, currentframe
from six.moves import xrange

from pyutilib.misc import flatten_list
from pyutilib.ply import t_newline, t_ignore, _find_column, p_error, ply_init
        

## -----------------------------------------------------------
##
## Lexer definitions for tokenizing the input
##
## -----------------------------------------------------------

_parse_info = None
debugging = False

states = (
  ('data','inclusive'),
)

reserved = {
    'data' : 'DATA',
    'set' : 'SET',
    'param' : 'PARAM',
    'end' : 'END',
    'store' : 'STORE',
    'load' : 'LOAD',
    'table' : 'TABLE',
    'include' : 'INCLUDE',
    'namespace' : 'NAMESPACE',
}

# Token names
tokens = [
    "COMMA",
    "LBRACE",
    "RBRACE",
    "SEMICOLON",
    "COLON",
    "COLONEQ",
    "LBRACKET",
    "RBRACKET",
    "LPAREN",
    "RPAREN",
    "WORD",
    "WORDWITHLBRACKET",
    "STRING",
    "BRACKETEDSTRING",
    "QUOTEDSTRING",
    "EQ",
    "TR",
    "ASTERISK",
    #"NONWORD",
    "INT_VAL",
    "FLOAT_VAL",
] + list(reserved.values())

# Regular expression rules
t_COMMA     = r","
t_LBRACKET  = r"\["
t_RBRACKET  = r"\]"
t_LBRACE  = r"\{"
t_RBRACE  = r"\}"
t_COLON     = r":"
t_EQ        = r"="
t_TR        = r"\(tr\)"
t_LPAREN    = r"\("
t_RPAREN    = r"\)"
t_ASTERISK  = r"\*"

# Discard comments
def t_COMMENT(t):
    r'(\#[^\n]*)|(/\*(.*?).?(\*/))'

def t_COLONEQ(t):
    r':='
    t.lexer.begin('data')
    return t

def t_SEMICOLON(t):
    r';'
    t.lexer.begin('INITIAL')
    return t

def t_WORDWITHLBRACKET(t):
    r'[a-zA-Z0-9_][a-zA-Z0-9_\.\-]*\['
    if t.value in reserved:
        t.type = reserved[t.value]    # Check for reserved words
    return t

def t_WORD(t):
    r'[a-zA-Z_0-9][a-zA-Z_0-9\.+\-]*'
    if t.value in reserved:
        t.type = reserved[t.value]    # Check for reserved words
    return t

def t_STRING(t):
    r'[a-zA-Z0-9_\.+\-]+'
    if t.value in reserved:
        t.type = reserved[t.value]    # Check for reserved words
    return t

def t_FLOAT_VAL(t):
    '[-+]?[0-9]+(\.([0-9]+)?([eE][-+]?[0-9]+)?|[eE][-+]?[0-9]+)'
    try:
        t.value = float(t.value)
        #t.type = "FLOAT_VAL"
        return t
    except:
        print("ERROR: "+t.value)
        raise IOError

def t_INT_VAL(t):
    '[-+]?[0-9]+([eE][-+]?[0-9]+)?'
    #t.type = "INT_VAL"
    t.value = int(t.value)
    return t

def t_data_BRACKETEDSTRING(t):
    r'[a-zA-Z0-9_\.+\-]*\[[a-zA-Z0-9_\.+\-\*,\s]+\]'
    # NO SPACES
    # a[1,_df,'foo bar']
    # [1,*,'foo bar']
    if t.value in reserved:
        t.type = reserved[t.value]    # Check for reserved words
    return t

def t_QUOTEDSTRING(t):
    r'"([^"]|\"\")*"|\'([^\']|\'\')*\''
    if t.value in reserved:
        t.type = reserved[t.value]    # Check for reserved words
    return t

#t_NONWORD   = r"[^\.A-Za-z0-9,;:=<>\*\(\)\#{}\[\] \n\t\r]+"

# Error handling rule
def t_error(t):             #pragma:nocover
    raise IOError("ERROR: Token %s Value %s Line %s Column %s" % (t.type, t.value, t.lineno, t.lexpos))
    t.lexer.skip(1)


## -----------------------------------------------------------
##
## Yacc grammar for data commands
##
## -----------------------------------------------------------

def p_expr(p):
    '''expr : statements
            | '''
    if len(p) == 2:
        #print "STMTS",p[1]
        for stmt in p[1]:
            if type(stmt) is list:
                _parse_info[None].append(stmt)
            else:
                for key in stmt:
                    if key in _parse_info:
                        _parse_info[key].append(stmt[key])
                    else:
                        _parse_info[key] = stmt[key]

def p_statements(p):
    '''statements : statements statement
                  | statement
                  | statements NAMESPACE WORD LBRACE statements RBRACE
                  | NAMESPACE WORD LBRACE statements RBRACE '''
    #print "STMT X",p[1:],p[1]
    len_p = len(p)
    if len_p == 3:
        # NB: statements will never be None, but statement *could* be None
        p[0] = p[1]
        if p[2] is not None:
            p[0].append(p[2])
    elif len_p == 2:
        if p[1] is None:
            p[0] = []
        else:
            p[0] = [p[1]]
    elif len_p == 7:
        # NB: statements will never be None
        p[0] = p[1]
        p[0].append({p[3]:p[5]})
    else:
        # NB: statements will never be None
        p[0] = [{p[2] : p[4]}]

def p_statement(p):
    '''statement : SET WORD COLONEQ datastar SEMICOLON
                 | SET WORDWITHLBRACKET args RBRACKET COLONEQ datastar SEMICOLON
                 | SET WORD COLON itemstar COLONEQ datastar SEMICOLON
                 | PARAM items COLONEQ datastar SEMICOLON
                 | TABLE items COLONEQ datastar SEMICOLON
                 | LOAD items SEMICOLON
                 | STORE items SEMICOLON
                 | INCLUDE WORD SEMICOLON
                 | INCLUDE QUOTEDSTRING SEMICOLON
                 | DATA SEMICOLON
                 | END SEMICOLON
    '''
    #print "STATEMENT",len(p), p[1:]
    stmt = p[1]
    if stmt == 'set':
        if p[2][-1] == '[':
            p[0] = ['set', p[2][:-1], '['] + flatten_list([p[i] for i in xrange(3,len(p)-1)])
        else:
            p[0] = flatten_list([p[i] for i in xrange(1,len(p)-1)])
    elif stmt == 'param':
        p[0] = flatten_list([p[i] for i in xrange(1,len(p)-1)])
    elif stmt == 'include':
        p[0] = [p[i] for i in xrange(1,len(p)-1)]
    elif stmt == 'load':
        p[0] = [p[1]]+ p[2]
    elif stmt == 'store':
        p[0] = [p[1]]+ p[2]
    elif stmt == 'table':
        p[0] = [p[1]]+ [p[2]] + [p[4]]
    else:
        # Not necessary, but nice to document how statement could end up None
        p[0] = None 
    #print(p[0])

def p_datastar(p):
    '''
    datastar : data
             |
    '''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = []

def p_data(p):
    '''
    data : data WORD
         | data STRING
         | data QUOTEDSTRING
         | data BRACKETEDSTRING
         | data SET
         | data TABLE
         | data PARAM
         | data INT_VAL
         | data FLOAT_VAL
         | data LPAREN
         | data RPAREN
         | data COMMA
         | data ASTERISK
         | WORD
         | STRING
         | QUOTEDSTRING
         | BRACKETEDSTRING
         | SET
         | TABLE
         | PARAM
         | INT_VAL
         | FLOAT_VAL
         | LPAREN
         | RPAREN
         | COMMA
         | ASTERISK
    '''
    # Locate and handle item as necessary
    single_item = len(p) == 2
    if single_item:
        tmp = p[1]
    else:
        tmp = p[2]
    if type(tmp) is str and tmp[0] == '"' and tmp[-1] == '"' and len(tmp) > 2 and not ' ' in tmp:
        tmp = tmp[1:-1]

    # Grow items list according to parsed item length
    if single_item:
        p[0] = [tmp]
    else:
        # yacc __getitem__ is expensive: use a local list to avoid a
        # getitem call on p[0]
        tmp_lst = p[1]
        tmp_lst.append(tmp)
        p[0] = tmp_lst

def p_args(p):
    '''
    args : arg
         |
    '''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = []

def p_arg(p):
    '''
    arg : arg COMMA WORD
         | arg COMMA STRING
         | arg COMMA QUOTEDSTRING
         | arg COMMA SET
         | arg COMMA TABLE
         | arg COMMA PARAM
         | arg COMMA INT_VAL
         | arg COMMA FLOAT_VAL
         | WORD
         | STRING
         | QUOTEDSTRING
         | SET
         | TABLE
         | PARAM
         | INT_VAL
         | FLOAT_VAL
    '''
    # Locate and handle item as necessary
    single_item = len(p) == 2
    if single_item:
        tmp = p[1]
    else:
        tmp = p[3]
    if type(tmp) is str and tmp[0] == '"' and tmp[-1] == '"' and len(tmp) > 2 and not ' ' in tmp:
        tmp = tmp[1:-1]

    # Grow items list according to parsed item length
    if single_item:
        p[0] = [tmp]
    else:
        # yacc __getitem__ is expensive: use a local list to avoid a
        # getitem call on p[0]
        tmp_lst = p[1]
        tmp_lst.append(tmp)
        p[0] = tmp_lst

def p_itemstar(p):
    '''
    itemstar : items
             |
    '''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = []

def p_items(p):
    '''
    items : items WORD
          | items STRING
          | items QUOTEDSTRING
          | items COMMA
          | items COLON
          | items LBRACE
          | items RBRACE
          | items LBRACKET
          | items RBRACKET
          | items TR
          | items LPAREN
          | items RPAREN
          | items ASTERISK
          | items EQ
          | items SET
          | items TABLE
          | items PARAM
          | items INT_VAL
          | items FLOAT_VAL
          | WORD
          | STRING
          | QUOTEDSTRING
          | COMMA
          | COLON
          | LBRACKET
          | RBRACKET
          | LBRACE
          | RBRACE
          | TR
          | LPAREN
          | RPAREN
          | ASTERISK
          | EQ
          | SET
          | TABLE
          | PARAM
          | INT_VAL
          | FLOAT_VAL
    '''
    # Locate and handle item as necessary
    single_item = len(p) == 2
    if single_item:
        tmp = p[1]
    else:
        tmp = p[2]
    if type(tmp) is str and tmp[0] == '"' and tmp[-1] == '"' and len(tmp) > 2 and not ' ' in tmp:
        tmp = tmp[1:-1]

    # Grow items list according to parsed item length
    if single_item:
        p[0] = [tmp]
    else:
        # yacc __getitem__ is expensive: use a local list to avoid a
        # getitem call on p[0]
        tmp_lst = p[1]
        tmp_lst.append(tmp)
        p[0] = tmp_lst


# --------------------------------------------------------------
# the DAT file lexer and yaccer only need to be
# created once, so have the corresponding objects
# accessible at module scope.
# --------------------------------------------------------------

tabmodule = 'parse_table_datacmds'

dat_lexer = None
dat_yaccer = None

#
# The function that performs the parsing
#
def parse_data_commands(data=None, filename=None, debug=0, outputdir=None):

    global debugging
    global dat_lexer
    global dat_yaccer

    if outputdir is None:
        # Try and write this into the module source...
        outputdir = os.path.dirname(getfile( currentframe() ))
        # Ideally, we would pollute a per-user configuration directory
        # first -- something like ~/.pyomo.
        if not os.access(outputdir, os.W_OK):
            outputdir = os.getcwd()

    # if the lexer/yaccer haven't been initialized, do so.
    if dat_lexer is None:
        #
        # Always remove the parser.out file, which is generated to
        # create debugging
        #
        if os.path.exists("parser.out"):        #pragma:nocover
            os.remove("parser.out")
        if debug > 0:                           #pragma:nocover
            #
            # Remove the parsetab.py* files.  These apparently need to
            # be removed to ensure the creation of a parser.out file.
            #
            if os.path.exists(tabmodule+".py"):
                os.remove(tabmodule+".py")
            if os.path.exists(tabmodule+".pyc"):
                os.remove(tabmodule+".pyc")
            debugging=True

        dat_lexer = lex.lex()
        #
        tmpsyspath = sys.path
        sys.path.append(outputdir)
        dat_yaccer = yacc.yacc(debug=debug, 
                                    tabmodule=tabmodule, 
                                    outputdir=outputdir,
                                    optimize=True)
        sys.path = tmpsyspath

    #
    # Initialize parse object
    #
    global _parse_info
    _parse_info = {}
    _parse_info[None] = []

    #
    # Parse the file
    #
    global _parsedata
    if not data is None:
        _parsedata=data
        ply_init(_parsedata)
        dat_yaccer.parse(data, lexer=dat_lexer, debug=debug)
    elif not filename is None:
        f = open(filename, 'r')
        try:
            data = f.read()
        except Exception:
            e = sys.exc_info()[1]
            f.close()
            del f
            raise e
        f.close()
        del f
        _parsedata=data
        ply_init(_parsedata)
        dat_yaccer.parse(data, lexer=dat_lexer, debug=debug)
    else:
        _parse_info = None
    #
    # Disable parsing I/O
    #
    debugging=False
    #print(_parse_info)
    return _parse_info

if __name__ == '__main__':
    parse_data_commands(filename=sys.argv[1], debug=100)
