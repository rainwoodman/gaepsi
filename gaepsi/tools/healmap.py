from healpy.projector import MollweideProj as MP
from healpy import vec2pix, npix2nside
from numpy import pi
projs = {'mollweide': MP}

def healmap(map, proj='mollweide', nest=False, rot=None, coord=None, flipconv=None):
  proj = projs[proj](flipconv=flipconv)
  f = lambda x,y,z: vec2pix(npix2nside(len(map)), x,y,z,nest)
  image = proj.projmap(map, f, rot=rot, coord=coord)
  return image, [v * 0.5 * pi for v in proj.get_extent()]
