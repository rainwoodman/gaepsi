import gaepsi._gaepsiccode as _ccode
from numpy import fromfile
from numpy import zeros, dtype, ones
from numpy import uint8, array, linspace
from numpy import int32, float32, int16
from numpy import log10

def circle(target, X, Y, V, scale, min=None, max=None, logscale=False, colormap=None): 
  if colormap == None:
    colormap = Colormap(levels = [0, 0.2, 0.4, 0.6, 1.0], 
                         r=[0, 0.5, 1.0, 1.0, 0.2], 
                         g=[0, 0.0, 0.5, 1.0, 0.2], 
                         b=[0.0, 0.0, 0.0, 0.3, 1.0])
  if len(target.shape) != 3:
     raise ValueError("has to be a rgb bitmap! expecting shape=(#,#,3)")
  image = target
  if min == None: min = V.min()
  if max == None: max = V.max()
  if min == max: 
    min = min - 0.5
    max = max + 0.5

  _ccode.circle(target = target, X = X, Y = Y, V = V,
    min = min, max = max,
    scale = scale,
    logscale=logscale,
    cmapr=colormap.table['r'],
    cmapg=colormap.table['g'],
    cmapb=colormap.table['b'],
    cmapa=colormap.table['a'],
    cmapv=colormap.table['v'])

def line (target, X, Y, VX,VY, scale, min=None, max=None, logscale=False, colormap=None): 
  if colormap == None:
    colormap = Colormap(levels = [0, 0.2, 0.4, 0.6, 1.0], 
                         r=[0, 0.5, 1.0, 1.0, 0.2], 
                         g=[0, 0.0, 0.5, 1.0, 0.2], 
                         b=[0.0, 0.0, 0.0, 0.3, 1.0])
  if len(target.shape) != 3:
     raise ValueError("has to be a rgb bitmap! expecting shape=(#,#,3)")
  image = target
#fix me: use the norm! how to avoid creating a new array?
  if min == None: min = array([VX.min(), VY.min()]).max()
  if max == None: max = array([VX.max(), VY.max()]).max()
  if min == max: 
    min = min - 0.5
    max = max + 0.5

  _ccode.line(target = target, X = X, Y = Y, VX = VX, VY=VY,
    min = min, max = max,
    scale = scale,
    logscale=logscale,
    cmapr=colormap.table['r'],
    cmapg=colormap.table['g'],
    cmapb=colormap.table['b'],
    cmapa=colormap.table['a'],
    cmapv=colormap.table['v'])

def color(target, raster, min, max, logscale=True, colormap=None):
  if colormap == None:
    colormap = Colormap(levels = [0, 0.2, 0.4, 0.6, 1.0], 
                         r=[0, 0.5, 1.0, 1.0, 0.2], 
                         g=[0, 0.0, 0.5, 1.0, 0.2], 
                         b=[0.0, 0.0, 0.0, 0.3, 1.0])

  _ccode.color(target=target, raster = raster, min = min, max = max, 
    logscale = logscale, 
    cmapr=colormap.table['r'],
    cmapg=colormap.table['g'],
    cmapb=colormap.table['b'],
    cmapa=colormap.table['a'])
  return target

