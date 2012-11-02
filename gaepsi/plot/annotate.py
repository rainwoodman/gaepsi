from numpy import empty
def annotate(field, xrange, yrange, zrange, npixels, scale):
  pos = field['locations'].copy()
  pos[:, 0] -= xrange[0]
  pos[:, 1] -= yrange[0]
  pos[:, 0] *= float(npixels[0]) / field.boxsize[0]
  pos[:, 1] *= float(npixels[1]) / field.boxsize[1]
  mask = (pos[:, 0] >= - scale)
  mask &= (pos[:, 0] <= (npixels[0]+scale))
  mask &= (pos[:, 1] >= (-scale))
  mask &= (pos[:, 1] <= (npixels[1]+scale))

  return pos[mask, 0], pos[mask, 1], field['default'][mask]


