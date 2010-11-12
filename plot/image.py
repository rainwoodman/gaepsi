import gadget
import gadget.ccode
import gadget.kernel

def image(field, xrange, yrange, zrange, npixels, quick=True):
    return gadget.ccode.image.image(field['locations'],
          field['sml'], field['default'],
          xrange[0], yrange[0], xrange[1], yrange[1],
          npixels[0], npixels[1], zrange[0], zrange[1], quick)
