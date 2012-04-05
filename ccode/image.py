import gaepsi._gaepsiccode as _ccode
from numpy import zeros
def rasterize(field, targets, values, xrange, yrange, zrange, quick=True):
  """rasterize the field listed in field to raster pixel arrays listed in targets. 
     field is a list of gadget.field.Field
     targets is a list of two dimensional arrays of the same shape
     the components are listed in values.
     returns the total number of particles in field contributed to the raster.
     """
  if type(targets) == list:
    if type(values) != list or len(values) != len(targets):
      raise ValueError('when targets is a list, values also have to be a matching list')
  else:
    targets = [targets]
    values = [values]
  Vs = [field[fieldname] for fieldname in values]
  expandedV = []
  expandedT = []
  for V,T in zip(Vs, targets):
    if len(V.shape) > 1:
      Vx = V[:, 0]
      Vy = V[:, 1]
      Vz = V[:, 2]
      Tx = T[:, :, 0]
      Ty = T[:, :, 1]
      Tz = T[:, :, 2]
      expandedV += [Vx,Vy,Vz]
      expandedT += [Tx,Ty,Tz]
    else:
      expandedV += [V]
      expandedT += [T]
  return _ccode.image(targets = expandedT, locations = field['locations'],
          sml = field['sml'], values = expandedV,
          xmin = xrange[0], ymin = yrange[0], xmax = xrange[1], ymax = yrange[1],
          zmin = zrange[0], zmax = zrange[1], mask = field.mask, quick = quick, boxsize=field.boxsize)

