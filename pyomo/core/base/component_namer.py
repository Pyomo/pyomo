#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.deprecation import deprecated, relocated_module_attribute
from pyomo.common.formatting import index_repr as _index_repr

for attr, new_attr in (
    ('literals', 'name_literals'),
    ('special_chars', 'name_special_chars'),
    ('re_number', 're_number'),
    ('re_special_char', 're_name_special_char'),
    ('name_repr', 'name_repr'),
    ('tuple_repr', 'tuple_repr'),
):
    relocated_module_attribute(
        attr,
        'pyomo.common.formatting.' + new_attr,
        version='6.10.0.dev0',
        f_globals=globals(),
    )


@deprecated(
    "index_repr has moved to pyomom.common.formatting.index_repr.  "
    "Note that the return value has also changed.",
    version='6.10.0.dev0',
)
def index_repr(idx, unknown_handler=str):
    return '[' + _index_repr(idx, unknown_handler) + ']'
