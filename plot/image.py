import gadget
import gadget.ccode
import gadget.kernel
from numpy import zeros

def image(field, xrange, yrange, zrange, npixels, quick=True, target=None):
  if target == None: 
    target = zeros(shape = npixels, dtype='f4')
  if target.shape[0] != npixels[0] or target.shape[1] != npixels[1]:
    raise ValueError("the shape of the target image is wrong")
  gadget.ccode.image.image(target, field['locations'],
          field['sml'], field['default'],
          xrange[0], yrange[0], xrange[1], yrange[1],
          npixels[0], npixels[1], zrange[0], zrange[1], quick)
  return target
