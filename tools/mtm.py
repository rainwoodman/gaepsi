from IPython.Debugger import Tracer
debug = Tracer()

from numpy import zeros, random, asarray, select, newaxis, fmin
def toss(x, p, sh=None):
  c = p.cumsum()
  c /= c[-1]
  y = random.random(sh)
  i = c.searchsorted(y)
  return x[i]

def mtm(weight, x0, step=1.0, skip=0, tries=1):
  shape = list(asarray(x0)[newaxis].shape)
  shape[0] = tries
  for i in range(-1, skip):
    delta = random.normal(scale=step, size=shape)
    y = delta - x0
    wy = weight(y)
    y0 = toss(y, p = wy)
    delta = random.normal(scale=step, size=shape)
    x = y0 - delta
    x[tries - 1] = x0
    wx = weight(x)
    wxsum = wx.sum()
    r = fmin(wxsum, wy.sum())
    if wxsum * random.uniform() < r: x0 = y0
  return x0

