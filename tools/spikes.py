from numpy import linspace, zeros, ones, newaxis, asarray, arange
from matplotlib.collections import LineCollection

from matplotlib.colors import colorConverter

def SpikeCollection(x, y, radius, linewidth=1, color=(1, 1, 0.9), alpha=0.9, gamma=3.33):
  """ returns a LineCollection of spikes.
      example:
      ax = gca()
      c = make_spike(x=[0, 5, -3, 3], y=[0, 2, -1, -5], radius=[10, 5, 5, 8], color=[(1, 1, 0.9), (0, 1, 0)], alpha=0.9)
      ax.add_collection(c)
      x, y, radius needs to be of the same length, otherwise the length of x is taken as the number of points
      color is repeated if it is shorter than the list of x, y radius
      alpha is the maximum alpha in the spike. gamma is used to correct for the buggy matplotlib transparency code
  """
  x = asarray(x).reshape(-1)
  y = asarray(y).reshape(-1)
  radius = asarray(radius).reshape(-1)

  Npoints = x.shape[0]

  alpha0 = 0.05
  Nseg = int(alpha / alpha0)
  alpha0 = alpha / Nseg

  l = linspace(0, 1, Nseg, endpoint=False)
  l **= 1 / gamma

  lines = zeros((Npoints, Nseg * 2, 2, 2))
  lines[:, :Nseg, 0, 1] = (-1 + l)
  lines[:, :Nseg, 1, 1] = (1. - l)

  lines[:, Nseg:, 0, 0] = (-1 + l)
  lines[:, Nseg:, 1, 0] = (1. - l)

  lines[:, :, :, :] *= radius[:, newaxis, newaxis, newaxis]
  lines[:, :, :, 0] += x[:, newaxis, newaxis]
  lines[:, :, :, 1] += y[:, newaxis, newaxis]

  lines.shape = -1, 2, 2

  colors = colorConverter.to_rgba_array(color).repeat(Nseg * 2, axis=0).reshape(-1, Nseg * 2, 4)
# this formular is math trick:
# it ensures the alpha value of the line segment N, when overplot on the previous line segment 0..N-1,
# gives the correct alpha_N = N * alpha0.
# solve (1 - x_N) (N-1) alpha_0 + x_N = N alpha_0
  colors[:, :Nseg, 3] = 1.0 / ( 1. / alpha0 - arange(Nseg))[newaxis, :]
  colors[:, Nseg:, 3] = colors[:, :Nseg, 3]
  colors.shape = -1, 4

  c = LineCollection(lines, linewidth, colors=colors, antialiased=1)
  return c

