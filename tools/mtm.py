from IPython.Debugger import Tracer
debug = Tracer()

from numpy import zeros, random, asarray, select
def toss(x, p, sh=None):
  c = p.cumsum()
  c /= c[-1]
  y = random.random(sh)
  i = c.searchsorted(y)
  return x[i]

def mtm(weight, x0):
  tries = 10
  shape = list(asarray(x0)[newaxis].shape)
  shape[0] = tries

  delta = random.normal(scale=1.0, size=shape)
  y = delta + x0
  wy = weight(y)
  y0 = toss(y, p = wy)
  delta = random.normal(scale=1.0, size=shape)
  x = y0 + delta
  x[tries - 1] = x0
  wx = weight(x)
  r = fmin(1, wy.sum() / wx.sum())
  #debug()
  if random.uniform() < r: return y0
  return x0

