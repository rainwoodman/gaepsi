import numpy
from numpy import ones

class KeyFrames:
  def __init__(self, specs, keyframes):
    self.frameid = 0
    self.paramspecs = specs
    self.keyframes = keyframes
    self.nframes = keyframes[-1][0]
    self.frames = [(i, dict()) for i in range(self.nframes)]

    for param in specs:
      if not hasattr(specs[param], '__call__'):
        self.fillparam(param)

  def convert(self, dt, value):
    if dt.type is not numpy.void:
      return dt.type(value)
    else:
      return numpy.asarray(value, dt.base)

  def fillparam(self, param):
    useful = []
    for frameid, keyframe in self.keyframes:
      if param in keyframe:
        useful += [(frameid, keyframe)]

    dt = numpy.dtype(self.paramspecs[param])
    
    for cursor in range(len(useful) -1):
      start, startf= useful[cursor]
      end, endf = useful[cursor+1]

      for frameid in range(start, end):
        x = (1.0 * (frameid - start)) / (end - start)
        dict = self.frames[frameid][1]
        v1 = self.convert(dt, endf[param])
        v0 = self.convert(dt, startf[param])
        # interpolate only if v1 and v0 are different otherwise
        # the strange numerical rounding offs mess it all up

        dict[param] = self.convert(dt, v1 * x + v0 * (1. - x))
        try:
          if v1 == v0: 
            dict[param] = v1
        except ValueError:
          if (v1 == v0).all():
            dict[param] = v1


  def move(self, frameid):
    self.frameid = frameid

  def __getattr__(self, name):
    v = self.paramspecs[name]
    try: return v(self)
    except TypeError:
      return self.frames[self.frameid][1][name]
