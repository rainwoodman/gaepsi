from gadget import ccode
# important to import kernel here otherwise the kernel arrays
# are not initialized.
import gadget.kernel
from numpy import zeros
def rasterize(target, field, xrange, yrange, zrange, quick=True):
  ccode.image.image(target, field['locations'],
          field['sml'], field['default'],
          xrange[0], yrange[0], xrange[1], yrange[1],
          zrange[0], zrange[1], quick)

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
  
