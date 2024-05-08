import pyomo.environ as pe
from .relaxations_base import BaseRelaxationData
from typing import Generator


def relaxation_data_objects(
    block, descend_into=True, active=None, sort=False
) -> Generator[BaseRelaxationData, None, None]:
    """
    Iterate over all instances of BaseRelaxationData in the block.

    Parameters
    ----------
    block: pyomo.core.base.block._BlockData
        The Block in which to look for relaxations
    descend_into: bool
        Whether or not to look for relaxations in sub-blocks
    active: bool
        If True, then any relaxations that have been deactivated or live on deactivated blocks will not be returned.
    sort: bool

    Returns
    -------
    relaxations: generator
        A generator yielding the relaxation objects.
    """
    for b in block.component_data_objects(
        pe.Block, descend_into=descend_into, active=active, sort=sort
    ):
        if isinstance(b, BaseRelaxationData):
            yield b


def _nonrelaxation_block_objects(block, descend_into=True, active=None, sort=False):
    for b in block.component_data_objects(
        pe.Block, descend_into=False, active=active, sort=sort
    ):
        if isinstance(b, BaseRelaxationData):
            continue
        else:
            yield b
            if descend_into:
                for _b in _nonrelaxation_block_objects(
                    b, descend_into=True, active=active, sort=sort
                ):
                    yield _b


def nonrelaxation_component_data_objects(
    block, ctype=None, active=None, sort=False, descend_into=True
):
    """
    Iterate over all components with the corresponding ctype (e.g., Constraint) in the block excluding
    those instances which are or live on relaxation objects (instances of BaseRelaxationData).

    Parameters
    ----------
    block: pyomo.core.base.block._BlockData
        The Block in which to look for components
    ctype: type
        The type of component to iterate over
    descend_into: bool
        Whether or not to look for components in sub-blocks
    active: bool
        If True, then any components that have been deactivated or live on deactivated blocks will not be returned.
    sort: bool

    Returns
    -------
    components: generator
        A generator yielding the requested components.
    """
    if not isinstance(ctype, type):
        raise ValueError(
            "nonrelaxation_component_data_objects expects ctype to be a type, not a "
            + str(type(ctype))
        )
    if ctype is pe.Block:
        for b in _nonrelaxation_block_objects(
            block, descend_into=descend_into, active=active, sort=sort
        ):
            yield b
    else:
        for comp in block.component_data_objects(
            ctype=ctype, descend_into=False, active=active, sort=sort
        ):
            yield comp
        if descend_into:
            for b in _nonrelaxation_block_objects(
                block, descend_into=True, active=active, sort=sort
            ):
                for comp in b.component_data_objects(
                    ctype=ctype, descend_into=False, active=active, sort=sort
                ):
                    yield comp
