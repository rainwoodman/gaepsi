from gadget import ccode
# important to import kernel here otherwise the kernel arrays
# are not initialized.
import gadget.kernel
from numpy import zeros
def rasterize(field, targets, values, xrange, yrange, zrange, quick=True):
  if type(targets) == list:
    if type(values) != list or len(values) != len(targets):
      raise ValueError('when targets is a list, values also have to be a matching list')
  else:
    targets = [targets]
    values = [values]
  V = [field[fieldname] for fieldname in values]
  
  ccode.image.image(targets = targets, locations = field['locations'],
          sml = field['sml'], values = V,
          xmin = xrange[0], ymin = yrange[0], xmax = xrange[1], ymax = yrange[1],
          zmin = zrange[0], zmax = zrange[1], quick = quick)

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
