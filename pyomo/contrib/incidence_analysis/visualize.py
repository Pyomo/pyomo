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
"""Module for visualizing results of incidence graph or matrix analysis"""
from pyomo.contrib.incidence_analysis.config import IncidenceOrder
from pyomo.contrib.incidence_analysis.interface import (
    IncidenceGraphInterface,
    get_structural_incidence_matrix,
)
from pyomo.common.dependencies import matplotlib


def _partition_variables_and_constraints(
    model, order=IncidenceOrder.dulmage_mendelsohn_upper, **kwds
):
    """Partition variables and constraints in an incidence graph"""
    igraph = IncidenceGraphInterface(model, **kwds)
    vdmp, cdmp = igraph.dulmage_mendelsohn()

    ucv = vdmp.unmatched + vdmp.underconstrained
    ucc = cdmp.underconstrained

    ocv = vdmp.overconstrained
    occ = cdmp.overconstrained + cdmp.unmatched

    ucvblocks, uccblocks = igraph.get_connected_components(
        variables=ucv, constraints=ucc
    )
    ocvblocks, occblocks = igraph.get_connected_components(
        variables=ocv, constraints=occ
    )
    wcvblocks, wccblocks = igraph.block_triangularize(
        variables=vdmp.square, constraints=cdmp.square
    )
    # By default, we block-*lower* triangularize. By default, however, we want
    # the Dulmage-Mendelsohn decomposition to be block-*upper* triangular.
    wcvblocks.reverse()
    wccblocks.reverse()
    vpartition = [ucvblocks, wcvblocks, ocvblocks]
    cpartition = [uccblocks, wccblocks, occblocks]

    if order == IncidenceOrder.dulmage_mendelsohn_lower:
        # If a block-lower triangular matrix was requested, we need to reverse
        # both the inner and outer partitions
        vpartition.reverse()
        cpartition.reverse()
        for vb in vpartition:
            vb.reverse()
        for cb in cpartition:
            cb.reverse()

    return vpartition, cpartition


def _get_rectangle_around_coords(ij1, ij2, linewidth=2, linestyle="-"):
    i1, j1 = ij1
    i2, j2 = ij2
    buffer = 0.5
    ll_corner = (min(i1, i2) - buffer, min(j1, j2) - buffer)
    width = abs(i1 - i2) + 2 * buffer
    height = abs(j1 - j2) + 2 * buffer
    rect = matplotlib.patches.Rectangle(
        ll_corner,
        width,
        height,
        clip_on=False,
        fill=False,
        edgecolor="orange",
        linewidth=linewidth,
        linestyle=linestyle,
    )
    return rect


def spy_dulmage_mendelsohn(
    model,
    *,
    incidence_kwds=None,
    order=IncidenceOrder.dulmage_mendelsohn_upper,
    highlight_coarse=True,
    highlight_fine=True,
    skip_wellconstrained=False,
    ax=None,
    linewidth=2,
    spy_kwds=None,
):
    """Plot sparsity structure in Dulmage-Mendelsohn order on Matplotlib axes

    This is a wrapper around the Matplotlib ``Axes.spy`` method for plotting
    an incidence matrix in Dulmage-Mendelsohn order, with coarse and/or fine
    partitions highlighted. The coarse partition refers to the under-constrained,
    over-constrained, and well-constrained subsystems, while the fine partition
    refers to block diagonal or block triangular partitions of the former
    subsystems.

    Parameters
    ----------

    model: ``ConcreteModel``
        Input model to plot sparsity structure of

    incidence_kwds: dict, optional
        Config options for ``IncidenceGraphInterface``

    order: ``IncidenceOrder``, optional
        Order in which to plot sparsity structure. Default is
        ``IncidenceOrder.dulmage_mendelsohn_upper`` for a block-upper triangular
        matrix. Set to ``IncidenceOrder.dulmage_mendelsohn_lower`` for a
        block-lower triangular matrix.

    highlight_coarse: bool, optional
        Whether to draw a rectangle around the coarse partition. Default True

    highlight_fine: bool, optional
        Whether to draw a rectangle around the fine partition. Default True

    skip_wellconstrained: bool, optional
        Whether to skip highlighting the well-constrained subsystem of the
        coarse partition. Default False

    ax: ``matplotlib.pyplot.Axes``, optional
        Axes object on which to plot. If not provided, new figure
        and axes are created.

    linewidth: int, optional
        Line width of for rectangle used to highlight. Default 2

    spy_kwds: dict, optional
        Keyword arguments for ``Axes.spy``

    Returns
    -------

    fig: ``matplotlib.pyplot.Figure`` or ``None``
        Figure on which the sparsity structure is plotted. ``None`` if axes
        are provided

    ax: ``matplotlib.pyplot.Axes``
        Axes on which the sparsity structure is plotted

    """
    plt = matplotlib.pyplot
    if incidence_kwds is None:
        incidence_kwds = {}
    if spy_kwds is None:
        spy_kwds = {}

    vpart, cpart = _partition_variables_and_constraints(model, order=order)
    vpart_fine = sum(vpart, [])
    cpart_fine = sum(cpart, [])
    vorder = sum(vpart_fine, [])
    corder = sum(cpart_fine, [])

    imat = get_structural_incidence_matrix(vorder, corder)
    nvar = len(vorder)
    ncon = len(corder)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    markersize = spy_kwds.pop("markersize", None)
    if markersize is None:
        # At 10000 vars/cons, we want markersize=0.2
        # At 20 vars/cons, we want markersize=10
        # We assume we want a linear relationship between 1/nvar
        # and the markersize.
        markersize = (10.0 - 0.2) / (1 / 20 - 1 / 10000) * (
            1 / max(nvar, ncon) - 1 / 10000
        ) + 0.2

    ax.spy(imat, markersize=markersize, **spy_kwds)
    ax.tick_params(length=0)
    if highlight_coarse:
        start = (0, 0)
        for i, (vblocks, cblocks) in enumerate(zip(vpart, cpart)):
            # Get the total number of variables/constraints in this part
            # of the coarse partition
            nv = sum(len(vb) for vb in vblocks)
            nc = sum(len(cb) for cb in cblocks)
            stop = (start[0] + nv - 1, start[1] + nc - 1)
            if not (i == 1 and skip_wellconstrained) and nv > 0 and nc > 0:
                # Regardless of whether we are plotting in upper or lower
                # triangular order, the well-constrained subsystem is at
                # position 1
                #
                # The get-rectangle function doesn't look good if we give it
                # an "empty region" to box.
                ax.add_patch(
                    _get_rectangle_around_coords(start, stop, linewidth=linewidth)
                )
            start = (stop[0] + 1, stop[1] + 1)

    if highlight_fine:
        # Use dashed lines to distinguish inner from outer partitions
        # if we are highlighting both
        linestyle = "--" if highlight_coarse else "-"
        start = (0, 0)
        for vb, cb in zip(vpart_fine, cpart_fine):
            stop = (start[0] + len(vb) - 1, start[1] + len(cb) - 1)
            # Note that the subset's we're boxing here can't be empty.
            ax.add_patch(
                _get_rectangle_around_coords(
                    start, stop, linestyle=linestyle, linewidth=linewidth
                )
            )
            start = (stop[0] + 1, stop[1] + 1)

    return fig, ax
