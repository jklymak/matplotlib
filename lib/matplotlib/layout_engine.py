"""
A module to define layout engines for Matplotlib.

Figures have a ``layout_engine`` property that holds a subclass of
`~.LayoutEngine` defined here (or *None* for no layout).  At draw time
``figure.get_layout_engine.execute()`` is called, the goal of which is usually
to rearrange Axes on the figure to produce a pleasing layout.  Note that
this could also be implimented as a ``draw`` callback, however when
printing we often disable the layout engine for the final draw, so it is
helpful to have a different callback mechanism. It is also useful to
know the layout engine while the figure is being created, in particular
to deal with colorbars, so having a dedicated property of the figure to
querry is useful.

Matplotlib has two native layout engines, ``tight_layout`` and
``constrained_layout``, which are implimented using this formalism
as `.TightLayoutEngine` and `.ConstrainedLayoutEngine`.  While layout
engines tend to be complicated, users and downstream libraries can now create
their own layout engine and connect it to a figure.
"""

import matplotlib._api as _api

from matplotlib._constrained_layout import do_constrained_layout
from matplotlib.tight_layout import (get_subplotspec_list,
                                     get_tight_layout_figure)
from matplotlib.backend_bases import _get_renderer
from contextlib import nullcontext


class LayoutEngine():
    """
    Base class for Matplotlib layout engines.

    A layout engine can be passed to a figure at instantiation or
    at any time with `~.figure.Figure.set_layout_engine`.  However, note
    note that layout engines affect the creation of colorbars, so
    `~.figure.Figure.set_layout_engine` should be called before any
    colorbars are created.

    Once attached to a figure, the layout engine ``execute`` function
    is called at draw time by `~.figure.Figure.draw`, providing a special
    draw-time hook.

    Currently, there are two properties of ``LayoutEngine`` classes that
    are consulted while manipulating the figure.  ``engine.colorbar_gridspec``
    tells `.Figure.colorbar` whether to make the axes using the gridspec
    method (see `.colorbar.make_axes_gridspec`) or not
    (see `.colorbar.make_axes`); for the native layout engines
    `.ConstrainedLayoutEngine` sets this to *False*.  The second property
    is ``engine.adjust_compatible`` that stops `.Figure.subplots_adjust` from
    being run if it is not compatible with the layout engine
    (`.ConstrainedLayoutEngine` sets this to *False* also).
    """
    def __init__(self, **kwargs):
        self._figure = None
        self.colorbar_gridspec = True
        self._params = {}
        self.adjust_compatible = False

    def set(self, **kwargs):
        raise NotImplementedError

    def get(self):
        return self._params

    def execute(self):
        raise NotImplementedError

    def set_figure(self, figure):
        """
        Set the figure for the layout manager.
        """
        if self._figure == figure:
            return
        if self._figure is None:
            self._figure = figure
        else:
            raise ValueError(
                'Layout managers cannot be attached to more than one figure')

    def get_figure(self):
        return self._figure


class TightLayoutEngine(LayoutEngine):
    """
    Implements the ``tight_layout`` geometry management.
    """

    def __init__(self, *, pad=1.08, h_pad=None, w_pad=None,
                 rect=(0, 0, 1, 1)):
        """
        Initialize tight_layout engine.

        Parameters
        ----------
        pad : float, 1.08
            Padding between the figure edge and the edges of subplots, as a
            fraction of the font size.
        h_pad, w_pad : float
            Padding (height/width) between edges of adjacent subplots.
            Defaults to *pad*.
        rect : tuple[float, float, float, float], optional
            (left, bottom, right, top) rectangle in normalized figure
            coordinates that the subplots (including labels)
            will fit into. Defaults to using the entire figure.
        """
        super().__init__()
        self.colorbar_gridspec = True
        self.adjust_compatible = True
        self.set(pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)

    def execute(self):
        """
        Execute tight_layout.

        This decides the subplot parameters given the padding that
        will allow the axes labels to not be covered by other labels
        and axes.

        See also: `.figure.Figure.tight_layout` and `.pyplot.tight_layout`.
        """
        fig = self._figure
        info = self._params
        subplotspec_list = get_subplotspec_list(fig.axes)
        if None in subplotspec_list:
            _api.warn_external("This figure includes Axes that are not "
                               "compatible with tight_layout, so results "
                               "might be incorrect.")
        renderer = _get_renderer(fig)
        with getattr(renderer, "_draw_disabled", nullcontext)():
            kwargs = get_tight_layout_figure(
                fig, fig.axes, subplotspec_list, renderer,
                pad=info['pad'], h_pad=info['h_pad'], w_pad=info['w_pad'],
                rect=info['rect'])
        if kwargs:
            fig.subplots_adjust(**kwargs)

    def set(self, *, pad=None, w_pad=None, h_pad=None, rect=None):
        todo = ['pad', 'w_pad', 'h_pad', 'rect']
        for td in todo:
            if locals()[td] is not None:
                self._params[td] = locals()[td]


class ConstrainedLayoutEngine(LayoutEngine):
    """
    Implements the ``constrained_layout`` geometry management.
    """

    def __init__(self, *, h_pad=None, w_pad=None,
                 hspace=None, wspace=None):
        """
        Initialize ``constrained_layout`` settings.

        Parameters
        ----------
        fig : `~.figure.Figure`
            Figure the layout engine will be used on.
        h_pad, w_pad : float
            Padding around the axes elements in figure-normalized units.
            Default to :rc:`figure.constrained_layout.h_pad` and
            :rc:`figure.constrained_layout.w_pad`.
        hspace, wspace : float
            Fraction of the figure to dedicate to space between the
            axes.  These are evenly spread between the gaps between the axes.
            A value of 0.2 for a three-column layout would have a space
            of 0.1 of the figure width between each column.
            If h/wspace < h/w_pad, then the pads are used instead.
            Default to :rc:`figure.constrained_layout.hspace` and
            :rc:`figure.constrained_layout.wspace`.
        """
        super().__init__()
        self.colorbar_gridspec = False
        self.adjust_compatible = False
        self.set(w_pad=w_pad, h_pad=h_pad, wspace=wspace, hspace=hspace)

    def execute(self):
        """
        Perform constrained_layout and move and resize axes accordingly.
        """
        width, height = self._figure.get_size_inches()
        # pads are relative to the current state of the figure...
        w_pad = self._params['w_pad'] / width
        h_pad = self._params['h_pad'] / height

        return do_constrained_layout(self._figure, w_pad=w_pad, h_pad=h_pad,
                                     wspace=self._params['wspace'],
                                     hspace=self._params['hspace'])

    def set(self, *, h_pad=None, w_pad=None,
            hspace=None, wspace=None):
        """
        Set the pads for constrained_layout.

        Parameters
        ----------
        h_pad, w_pad : float
            Padding around the axes elements in figure-normalized units.
            Default to :rc:`figure.constrained_layout.h_pad` and
            :rc:`figure.constrained_layout.w_pad`.
        hspace, wspace : float
            Fraction of the figure to dedicate to space between the
            axes.  These are evenly spread between the gaps between the axes.
            A value of 0.2 for a three-column layout would have a space
            of 0.1 of the figure width between each column.
            If h/wspace < h/w_pad, then the pads are used instead.
            Default to :rc:`figure.constrained_layout.hspace` and
            :rc:`figure.constrained_layout.wspace`.
        """
        for td in self.set.__kwdefaults__:
            if locals()[td] is not None:
                self._params[td] = locals()[td]
