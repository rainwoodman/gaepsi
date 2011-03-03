from gadget import ccode

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
  return ccode.image(targets = expandedT, locations = field['locations'],
          sml = field['sml'], values = expandedV,
          xmin = xrange[0], ymin = yrange[0], xmax = xrange[1], ymax = yrange[1],
          zmin = zrange[0], zmax = zrange[1], mask = field.mask, quick = quick, boxsize=field.boxsize)

unusedcode= """
def sparse(field, xrange, yrange, zrange, scale):
  pos = field['locations'].copy()
  pos[:, 0] -= xrange[0]
  pos[:, 1] -= yrange[0]
  pos[:, 0] *= float(npixels[0]) / field.boxsize[0]
  pos[:, 1] *= float(npixels[1]) / field.boxsize[1]
  mask = (pos[:, 0] >= - scale)
  mask &= (pos[:, 0] <= (npixels[0]+scale))
  mask &= (pos[:, 1] >= (-scale))
  mask &= (pos[:, 1] <= (npixels[1]+scale))

  target = zeros(dtype = [('X', 'f4'), ('Y', 'f4'), ('V', 'f4')], shape = sum(mask))
  target['X'] = pos[mask, 0]
  target['Y'] = pos[mask, 1]
  target['V'] = field['default'][mask]
  return target
""" 
