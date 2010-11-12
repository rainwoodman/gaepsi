from numpy import empty
def annotate(field, xrange, yrange, zrange, npixels):
  pos = field['locations'].copy()
  mask = pos[:, 0] >= xrange[0]
  mask &= (pos[:, 0] <= xrange[1])
  pos[:, 0] -= xrange[0]
  pos[:, 1] -= yrange[0]
  pos[:, 0] *= float(npixels[0]) / field.boxsize[0]
  pos[:, 1] *= float(npixels[1]) / field.boxsize[1]

  return pos[mask, 0], pos[mask, 1], field['default'][mask]


