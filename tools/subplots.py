import numpy as np
import itertools

def subplots(nrows=1, ncols=1, gridspec=None, fig=None, sharex=False, sharey=False, squeeze=True,
                subplot_kw=None, **fig_kw):
    """Create a figure with a set of subplots already made.

    This utility wrapper makes it convenient to create common layouts of
    subplots, including the enclosing figure object, in a single call.

    Keyword arguments:

    figure: Figure to add subplots to.

    nrows : int
      Number of rows of the subplot grid.  Defaults to 1.

    ncols : int
      Number of columns of the subplot grid.  Defaults to 1.

    sharex : string
      If 'all', the X axis will be shared amongst all subplots.  
      If 'column', the X axis will be shared within columns.
      If not None or 'none', and you have multiple rows, the x tick labels on all but
      the last row of plots will have visible set to False

    sharey : string
      If 'all', the Y axis will be shared amongst all subplots. 
      If 'row', the Y axis will be shared within rows. 
      If not None or 'none' and you have multiple columns, the y tick labels on all but
      the first column of plots will have visible set to False

    squeeze : bool
      If True, extra dimensions are squeezed out from the returned axis object:

        - if only one subplot is constructed (nrows=ncols=1), the
          resulting single Axis object is returned as a scalar.

        - for Nx1 or 1xN subplots, the returned object is a 1-d numpy
          object array of Axis objects are returned as numpy 1-d
          arrays.

        - for NxM subplots with N>1 and M>1 are returned as a 2d
          array.

      If False, no squeezing at all is done: the returned axis object is always
      a 2-d array contaning Axis instances, even if it ends up being 1x1.

    subplot_kw : dict
      Dict with keywords passed to the add_subplot() call used to create each
      subplots.

    Returns:

      fig, ax
      - ax: can be either a single axis object or an array of axis
        objects if more than one supblot was created.  The dimensions
        of the resulting array can be controlled with the squeeze
        keyword, see above.

    **Examples:**
    """

    if subplot_kw is None: subplot_kw = {}
    if sharex == 'none': sharex = None
    if sharey == 'none': sharey = None

    if fig is None: 
        from pylab import figure
        fig = figure(**fig_kw)

    # Create empty object array to hold all axes.  It's easiest to make it 1-d
    # so we can just append subplots upon creation, and then
    if gridspec: nrows, ncols = gridspec.get_geometry()

    nplots = nrows*ncols
    axarr = np.empty((nrows, ncols), dtype=object)

    # Create first subplot separately, so we can share it if requested
    for i,j in itertools.product(range(nrows), range(ncols)):
        kw = subplot_kw.copy()
        if sharex == 'all':
            if i != 0 or j != 0:
                kw['sharex'] = axarr[0, 0]
        elif sharex == 'column':
            if i != 0:
                kw['sharex'] = axarr[0, j]
        if sharey == 'all':
            if i != 0 or j != 0:
                kw['sharey'] = axarr[0, 0]
        elif sharey == 'row':
            if j != 0:
                kw['sharey'] = axarr[i, 0]

        # Note off-by-one counting because add_subplot uses the MATLAB 1-based
        # convention.
        if gridspec:
            print i, j, kw
            axarr[i,j] = fig.add_subplot(gridspec[i, j], **kw)
        else:
            axarr[i,j] = fig.add_subplot(nrows, ncols, i * ncols + j + 1, **kw)
        
    # turn off redundant tick labeling
    if sharex and nrows>1:
        # turn off all but the bottom row
        for ax in axarr[:-1,:].flat:
            for label in ax.get_xticklabels():
                label.set_visible(False)

    if sharey and ncols>1:
        # turn off all but the first column
        for ax in axarr[:,1:].flat:
            for label in ax.get_yticklabels():
                label.set_visible(False)

    if squeeze:
        # Reshape the array to have the final desired dimension (nrow,ncol),
        # though discarding unneeded dimensions that equal 1.  If we only have
        # one subplot, just return it instead of a 1-element array.
        if nplots==1:
            ret = fig, axarr[0,0]
        else:
            ret = fig, axarr.squeeze()
    else:
        # returned axis array will be always 2-d, even if nrows=ncols=1
        ret = fig, axarr

    return ret

from scipy.optimize import curve_fit

def plot_with_fit(ax, x, y, symbol, color, label, fitfunc, p0=None):
  ax.plot(x, y, symbol, label=label, color=color)
  s = x.argsort()
  try: 
    p, err = curve_fit(fitfunc, x, y, p0)
    return ax.plot(x[s], fitfunc(x, *p)[s], ':', color=color), (p, err)
  except RuntimeError:
    return ax.plot(x[s], fitfunc(x, *p0)[s], ':', color=color), (p0, None)

def plot_without_fit(ax, x, y, symbol, color, label, fitfunc):
  s = x.argsort()
  return ax.plot(x[s], y[s], symbol, label=label, color=color), (None, None)
