import numpy

class Timeline:
  """
     tl = Timeline(dtype=
         [('pos', ('f4', 3)), ('theta', 'f4'), ('phi', 'f4'), ('d', 'f4')])
     tl.add(....)
  """
  def __init__(self, dtype):
    self.keyframes = {}
    self.dtype = numpy.dtype(dtype)
    self.timelines = {}
    self.fps = {}
  def add(self, time, **kwargs):
    if time not in self.keyframes:
      self.keyframes[time] = {}

    self.keyframes[time].update(kwargs)

  def compile(self):
    for key in self.dtype.names:
      times=[]
      values=[]
      for time in self.keyframes:
        if key in self.keyframes[time]:
          times += [time]
          values += [self.keyframes[time][key]]
      if len(times) and len(values):
        arg = numpy.argsort(numpy.asarray(times))
        self.timelines[key] = (numpy.asarray(times)[arg], numpy.asarray(values)[arg])
      else:
        self.timelines[key] = ([], [])

  def __getitem__(self, index):
    """ returns frames at times given by index """
    result = numpy.zeros_like(index, dtype=self.dtype)
    for key in self.dtype.names:
      times, values = self.timelines[key]
      if len(times) and len(values):
        result[key] = interp(index, times, values)
    return result

def interp(x, xp, yp, left=None, right=None):
  """ this will interpolate struct dtypes and 2-d arrays, too """
  assert len(yp) == len(xp)
  yp = numpy.asarray(yp)
  if left is None:
    left = yp[0]
  if right is None:
    right = yp[-1]
  if yp.dtype.fields:
    result = numpy.empty_like(x, yp.dtype)
    for key in yp.dtype.fields:
      result[key] = interp(x, xp, yp[key], left[key], right[key])
  elif len(yp.shape) > 1:
    result = numpy.empty_like(x, dtype=(yp.dtype.base, yp.shape[1:]))
    for key in range(yp.shape[1]):
      result[..., key] = interp(x, xp, yp[:, key], left[key], right[key])
  else:
    result = numpy.interp(x, xp, yp, left, right)
  return result

if False:
  f = Timeline([('pos', ('f4', 3)), ('mass', 'f4')])

  f.add(10, mass=1.)
  f.add(20, mass=2.)
  f.add(30, mass=3.)
  f.add(10, pos=[1, 2, 3.])
  f.add(20, pos=[2, 4, 6.])
  f.add(30, pos=[3, 6, 9.])
  f.compile()
  
  r = f[numpy.arange(10, 30, 5)]
  print r, r.dtype

  dt = numpy.dtype([('pos', ('f4', 3)), ('mass', 'f4')])
  yp = numpy.empty(10, dtype=dt)
  yp['pos'][:, 0] = numpy.arange(10) * 1
  yp['pos'][:, 1] = 0
  yp['pos'][:, 2] = numpy.arange(10) * 3
  yp['mass'][:] = numpy.arange(10) * 10
  xp = numpy.arange(10)
  print interp([3.5], xp, yp)
