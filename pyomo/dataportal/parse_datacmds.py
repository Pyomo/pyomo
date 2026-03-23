# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import bisect
import importlib
import pyomo.tpl.ply.lex as lex
import pyomo.tpl.ply.yacc as yacc

from pyomo.common.errors import DeveloperError
from pyomo.common.fileutils import this_file, this_file_dir
from pyomo.core.base.util import flatten_tuple

_re_number = r'[-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?'

## -----------------------------------------------------------
##
## Lexer definitions for tokenizing the input
##
## -----------------------------------------------------------

_parse_info = None

states = (('data', 'inclusive'),)

reserved = {
    'data': 'DATA',
    'set': 'SET',
    'param': 'PARAM',
    'end': 'END',
    'store': 'STORE',
    'load': 'LOAD',
    'table': 'TABLE',
    'include': 'INCLUDE',
    'namespace': 'NAMESPACE',
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
    "NUM_VAL",
    # "NONWORD",
] + list(reserved.values())

# Ignore space and tab
t_ignore = " \t\r"

# Regular expression rules
t_COMMA = r","
t_LBRACKET = r"\["
t_RBRACKET = r"\]"
t_LBRACE = r"\{"
t_RBRACE = r"\}"
t_COLON = r":"
t_EQ = r"="
t_TR = r"\(tr\)"
t_LPAREN = r"\("
t_RPAREN = r"\)"
t_ASTERISK = r"\*"


#
# Notes on PLY tokenization
#   - token functions (beginning with "t_") are prioritized in the order
#     that they are declared in this module
#   - use @lex.TOKEN instead of docstrings to avoid errors from the
#     Sphinx autosummary
#
@lex.TOKEN(r'[\n]+')
def t_newline(t):
    t.lexer.lineno += len(t.value)
    t.lexer.linepos.extend(t.lexpos + i for i, _ in enumerate(t.value))


# Discard comments
_re_singleline_comment = r'(?:\#[^\n]*)'
_re_multiline_comment = r'(?:/\*(?:[\n]|.)*?\*/)'


@lex.TOKEN('|'.join([_re_singleline_comment, _re_multiline_comment]))
def t_COMMENT(t):
    # Single-line and multi-line strings
    nlines = t.value.count('\n')
    t.lexer.lineno += nlines
    # We will never need to determine column numbers within this comment
    # block, so it is sufficient to just worry about the *last* newline
    # in the comment
    lastpos = t.lexpos + t.value.rfind('\n')
    t.lexer.linepos.extend(lastpos for i in range(nlines))


@lex.TOKEN(r':=')
def t_COLONEQ(t):
    t.lexer.begin('data')
    return t


@lex.TOKEN(r';')
def t_SEMICOLON(t):
    t.lexer.begin('INITIAL')
    return t


# Numbers must be followed by a delimiter token (EOF is not a concern,
# as valid DAT files always end with a ';').
@lex.TOKEN(_re_number + r'(?=[\s()\[\]{}:;,])')
def t_NUM_VAL(t):
    _num = float(t.value)
    if '.' in t.value:
        t.value = _num
    else:
        _int = int(_num)
        t.value = _int if _num == _int else _num
    return t


@lex.TOKEN(r'[a-zA-Z_][a-zA-Z0-9_\.\-]*\[')
def t_WORDWITHLBRACKET(t):
    return t


@lex.TOKEN(r'[a-zA-Z_][a-zA-Z_0-9\.+\-]*')
def t_WORD(t):
    if t.value in reserved:
        t.type = reserved[t.value]  # Check for reserved words
    return t


@lex.TOKEN(r'[a-zA-Z0-9_\.+\-\\\/]+')
def t_STRING(t):
    # Note: RE guarantees the string has no embedded quotation characters
    t.value = '"' + t.value + '"'
    return t


@lex.TOKEN(r'[a-zA-Z0-9_\.+\-]*\[[a-zA-Z0-9_\.+\-\*,\s]+\]')
def t_data_BRACKETEDSTRING(t):
    # NO SPACES
    # a[1,_df,'foo bar']
    # [1,*,'foo bar']
    return t


_re_quoted_str = r'"(?:[^"]|"")*"'


@lex.TOKEN("|".join([_re_quoted_str, _re_quoted_str.replace('"', "'")]))
def t_QUOTEDSTRING(t):
    # Normalize the quotes to use '"', and replace doubled ("escaped")
    # quotation characters with a single character
    t.value = '"' + t.value[1:-1].replace(2 * t.value[0], t.value[0]) + '"'
    return t


# t_NONWORD   = r"[^\.A-Za-z0-9,;:=<>\*\(\)\#{}\[\] \n\t\r]+"


# Error handling rule
def t_error(t):
    raise IOError(
        "ERROR: Token %s Value %s Line %s Column %s"
        % (t.type, t.value, t.lineno, t.lexpos)
    )


## DEBUGGING: uncomment to get tokenization information
# def _wrap(_name, _fcn):
#     def _wrapper(t):
#         print(_name + ": %s" % (t.value,))
#         return _fcn(t)
#     _wrapper.__doc__ = _fcn.__doc__
#     return _wrapper
# import inspect
# for _name in list(globals()):
#     if _name.startswith('t_') and inspect.isfunction(globals()[_name]):
#         globals()[_name] = _wrap(_name, globals()[_name])


def _lex_token_position(t):
    i = bisect.bisect_left(t.lexer.linepos, t.lexpos)
    if i:
        return t.lexpos - t.lexer.linepos[i - 1]
    return t.lexpos


## -----------------------------------------------------------
##
## Yacc grammar for data commands
##
## -----------------------------------------------------------


def p_expr(p):
    '''expr : statements
    |'''
    if len(p) == 2:
        # print "STMTS",p[1]
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
    | NAMESPACE WORD LBRACE statements RBRACE'''
    # print "STMT X",p[1:],p[1]
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
        p[0].append({p[3]: p[5]})
    else:
        # NB: statements will never be None
        p[0] = [{p[2]: p[4]}]


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
    # print "STATEMENT",len(p), p[1:]
    stmt = p[1]
    if stmt == 'set':
        if p[2][-1] == '[':
            # Just turn off the flatten_list and see what happens
            p[0] = ['set', p[2][:-1], '['] + list(
                flatten_tuple([p[i] for i in range(3, len(p) - 1)])
            )
        else:
            p[0] = list(flatten_tuple([p[i] for i in range(1, len(p) - 1)]))
    elif stmt == 'param':
        p[0] = list(flatten_tuple([p[i] for i in range(1, len(p) - 1)]))
    elif stmt == 'include':
        p[0] = [p[i] for i in range(1, len(p) - 1)]
    elif stmt == 'load':
        p[0] = [p[1]] + p[2]
    elif stmt == 'store':
        p[0] = [p[1]] + p[2]
    elif stmt == 'table':
        p[0] = [p[1]] + [p[2]] + [p[4]]
    else:
        # Not necessary, but nice to document how statement could end up None
        p[0] = None
    # print(p[0])


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
    data : data NUM_VAL
         | data WORD
         | data STRING
         | data QUOTEDSTRING
         | data BRACKETEDSTRING
         | data SET
         | data TABLE
         | data PARAM
         | data LPAREN
         | data RPAREN
         | data COMMA
         | data ASTERISK
         | NUM_VAL
         | WORD
         | STRING
         | QUOTEDSTRING
         | BRACKETEDSTRING
         | SET
         | TABLE
         | PARAM
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
    # if type(tmp) is str and tmp[0] == '"' and tmp[-1] == '"' and len(tmp) > 2 and not ' ' in tmp:
    #    tmp = tmp[1:-1]

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
    arg : arg COMMA NUM_VAL
         | arg COMMA WORD
         | arg COMMA STRING
         | arg COMMA QUOTEDSTRING
         | arg COMMA SET
         | arg COMMA TABLE
         | arg COMMA PARAM
         | NUM_VAL
         | WORD
         | STRING
         | QUOTEDSTRING
         | SET
         | TABLE
         | PARAM
    '''
    # Locate and handle item as necessary
    single_item = len(p) == 2
    if single_item:
        tmp = p[1]
    else:
        tmp = p[3]
    if (
        type(tmp) is str
        and tmp[0] == '"'
        and tmp[-1] == '"'
        and len(tmp) > 2
        and not ' ' in tmp
    ):
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
    items : items NUM_VAL
          | items WORD
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
          | NUM_VAL
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
    '''
    # Locate and handle item as necessary
    single_item = len(p) == 2
    if single_item:
        tmp = p[1]
    else:
        tmp = p[2]
    if (
        type(tmp) is str
        and tmp[0] == '"'
        and tmp[-1] == '"'
        and len(tmp) > 2
        and not ' ' in tmp
    ):
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


def p_error(p):
    if p is None:
        tmp = "Syntax error at end of file."
    else:
        tmp = "Syntax error at token '%s' with value '%s' (line %s, column %s)" % (
            p.type,
            p.value,
            p.lineno,
            _lex_token_position(p),
        )
    raise IOError(tmp)


# --------------------------------------------------------------
# the DAT file lexer and yaccer only need to be
# created once, so have the corresponding objects
# accessible at module scope.
# --------------------------------------------------------------


tabmodule = '_parse_table_datacmds'

dat_lexer = None
dat_yaccer = None


def _get_this_file_signature():
    """Compute the SHA-256 signature for this module.

    We store the signature in the auto-generated parse table module so
    that we can robustly detect when the parse tables must be
    regenerated.

    """
    import hashlib

    sha = hashlib.sha256()
    with open(this_file(), 'r') as FILE:
        sha.update(FILE.read().encode())
    return sha.hexdigest()


#
# The function that performs the parsing
#
def parse_data_commands(data=None, filename=None, debug=0):
    global dat_lexer
    global dat_yaccer

    # if the lexer/yaccer haven't been initialized, do so.
    if dat_lexer is None or dat_yaccer is None:
        # Import the parse table, and check that it is valid for this module
        pkg = __name__.rsplit('.', maxsplit=1)[0]
        dat_tabmodule = importlib.import_module('.' + tabmodule, pkg)
        if _get_this_file_signature() != dat_tabmodule._lr_module_signature:
            raise DeveloperError(
                f"DAT parse tables ({pkg}.{tabmodule}) out of sync with "
                f"parser definition; regenerate by running {__file__}"
            )

        dat_lexer = lex.lex()
        dat_yaccer = yacc.yacc(tabmodule=dat_tabmodule, debug=debug)

    #
    # Load the text to parse
    #
    if filename is not None:
        if data is not None:
            raise ValueError(
                "parse_data_commands: cannot specify both "
                "data and filename arguments"
            )
        with open(filename, 'r') as FILE:
            data = FILE.read()

    if data is None:
        return None

    #
    # Initialize the lexer / parser objects
    #
    global _parse_info
    _parse_info = {None: []}
    dat_lexer.linepos = []
    #
    # Parse
    #
    dat_yaccer.parse(data, lexer=dat_lexer, debug=debug)
    #
    # Reset parse data and return the result
    #
    result = _parse_info
    _parse_info = None
    dat_lexer.linepos = []
    return result


if __name__ == '__main__':  # pragma:nocover
    print(f"Regenerating {tabmodule}")
    yacc.yacc(
        tabmodule=tabmodule,
        outputdir=this_file_dir(),
        debug=100,
        module_signature=_get_this_file_signature(),
    )
