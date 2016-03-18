# for IO bound large rendering.
from gaepsi.compiledbase.camera import Camera, setupsml
import numpy
import sharedmem
from gaepsi.compiledbase import ztree as zt
from compiledbase import fillingcurve as fc
from matplotlib import cm
from gaepsi.tools import nl_, n_
def paint(pos, color, luminosity, sml, camera, CCD, tree=None,
        return_tree_and_sml=False, normalize=True, np=None, direct_write=False,
        cumulative=True):
    """ pos = (x, y, z)
        color can be None or 1 or array
        luminosity can be None or 1 or array
        sml can be None or 1, or array
        camera is Camera
        CCD is array of camera.shape, 2.
        CCD[..., 0] is the color channel (sum of color * luminosity)
        CCD[..., 1] is the luminosity channel (sum of luminosity)

        if color is None, CCD.shape == camera.shape
        if color is not None, 
            CCD.shape == camera.shape, 2
            CCD[..., 0] is color
            CCD[..., 1] is brightness

        if normalize is False, do not do CCD[..., 0] will be
        the weighted sum of color.
        if normalize is True, CCD[..., 0] will be the weighted average of color

        if direct_write is true, each process will directly write to CCD (CCD
        must be on sharedmem)

        if cumulative is False, original content in CCD will be disregarded.
        if cumulative is True, original content in CCD will be preserved (+=)
    """
    CCDlimit = 20 * 1024 * 1024 # 20M pixel per small CCD
    camera.shape = (CCD.shape[0], CCD.shape[1])
    nCCD = int((CCD.shape[0] * CCD.shape[1] / CCDlimit) ** 0.5)
    if np is None: np = sharedmem.cpu_count()
    if nCCD <= np ** 0.5: 
        nCCD = int(np ** 0.5 + 1)
    cams = camera.divide(nCCD, nCCD)
    cams = cams.reshape(-1, 3)

    if tree is None:
        scale = fc.scale([x.min() for x in pos], [x.ptp() for x in pos])
        zkey = sharedmem.empty(len(pos[0]), dtype=fc.fckeytype)

        with sharedmem.MapReduce(np=np) as pool:
            chunksize = 1024 * 1024
            def work(i):
                sl = slice(i, i+chunksize)        
                x, y, z = pos
                fc.encode(x[sl], y[sl], z[sl], scale=scale, out=zkey[i:i+chunksize])
            pool.map(work, range(0, len(zkey), chunksize))

        arg = numpy.argsort(zkey)

        tree = zt.Tree(zkey=zkey, scale=scale, arg=arg, minthresh=8, maxthresh=20)
    if sml is None:
        sml = sharedmem.empty(len(zkey), 'f4')
        with sharedmem.MapReduce(np=np) as pool:
            chunksize = 1024 * 64
            def work(i):
                setupsml(tree, [x[i:i+chunksize] for x in pos],
                        out=sml[i:i+chunksize])
            pool.map(work, range(0, len(zkey), chunksize))

    def writeCCD(i, sparse, myCCD):
        cam, ox, oy = cams[i]
        #print i, sparse, len(cams)
        if sparse:
            index, C, L = myCCD
            x = index[0] + ox
            y = index[1] + oy
            p = CCD.flat
            if color is not None:
                ind = numpy.ravel_multi_index((x, y, 0), CCD.shape)
                if cumulative:
                    p[ind] += C
                else:
                    p[ind] = C
                ind = numpy.ravel_multi_index((x, y, 1), CCD.shape)
                if cumulative:
                    p[ind] += L
                else:
                    p[ind] = L
            else:
                ind = numpy.ravel_multi_index((x, y), CCD.shape)
                if cumulative:
                    p[ind] += L
                else:
                    p[ind] = L
        else:
            if color is not None:
                if cumulative:
                    CCD[ox:ox + cam.shape[0], oy:oy+cam.shape[1], :] += myCCD
                else:
                    CCD[ox:ox + cam.shape[0], oy:oy+cam.shape[1], :] = myCCD
            else:
                if cumulative:
                    CCD[ox:ox + cam.shape[0], oy:oy+cam.shape[1]] += myCCD[..., 1]
                else:
                    CCD[ox:ox + cam.shape[0], oy:oy+cam.shape[1]] = myCCD[..., 1]

    with sharedmem.MapReduce(np=np) as pool:
        def work(i):
            cam, ox, oy = cams[i]
            myCCD = numpy.zeros(cam.shape, dtype=('f8', 2))
            cam.paint(pos[0], pos[1], pos[2], 
                    sml, color, luminosity, out=myCCD, tree=tree)
            mask = (myCCD[..., 1] != 0)
            if mask.sum() < 0.1 * myCCD[..., 1].size:
                index = mask.nonzero()
                C = myCCD[..., 0][mask]
                L = myCCD[..., 1][mask]
                sparse, myCCD = True, (index, C, L)
            else:
                sparse, myCCD = False, myCCD
            if not direct_write:
                return i, sparse, myCCD
            else:
                writeCCD(i, sparse, myCCD)
                return 0, 0, 0
        def reduce(i, sparse, myCCD):
            if not direct_write:
                writeCCD(i, sparse, myCCD)
        pool.map(work, range(len(cams)), reduce=reduce)

    if color is not None and normalize:
        CCD[..., 0] /= CCD[..., 1]
    if return_tree_and_sml:
        return CCD, tree, sml
    else:
        tree = None
        sml = None
        return CCD

def image(color, luminosity=None, cmap=None, composite=False):
    """converts (color, luminosity) to an rgba image,
       if composite is False, directly reduce luminosity
       on the RGBA channel by luminosity,
       if composite is True, reduce the alpha channel.
       color and luminosity needs to be normalized/clipped to (0, 1), 
       otherwise will give nonsense results.
    """
    if cmap is None:
      if luminosity is not None:
        cmap = cm.coolwarm
      else:
        cmap = cm.gist_heat
    image = cmap(color, bytes=True)
    if luminosity is not None:
      luminosity = luminosity.copy()
      if composite:
        numpy.multiply(luminosity, 255.999, image[..., 3], casting='unsafe')
      else:
        numpy.multiply(image[...,:3], luminosity[..., None], image[..., :3], casting='unsafe')
    return image


def addspikes(image, x, y, s, color):
  """deprecated method to directly add spikes to a raw image. 
     use context.scatter(fancy=True) instead
     s is pixels
  """
  color = numpy.array(colorConverter.to_rgba(color))
  for X, Y, S in zip(x, y, s):
    def work(image, XY, S, axis):
      S = int(S)
      if S < 1: return
      XY = numpy.int32(XY)
      start = XY[axis] - S
      end = XY[axis] + S + 1

      fga = ((1.0 - numpy.abs(numpy.linspace(-1, 1, end-start, endpoint=True))) * color[3])
      #pre multiply
      fg = color[0:3][None, :] * fga[:, None]

      if start < 0: 
        fga = fga[-start:]
        fg = fg[-start:]
        start = 0
      if end >= image.shape[axis]:
        end = image.shape[axis] - 1
        fga = fga[:end - start]
        fg = fg[:end - start]
 
      if axis == 0:
        sl = image[start:end, XY[1], :]
      elif axis == 1:
        sl = image[XY[0], start:end, :]

      bg = sl / 256.

      #bga = bg[..., 3]
      bga = 1.0
      #bg = bg[..., 0:3] * bga[..., None]
      c = bg * (1-fga[:, None]) + fg
      ca = (bga * (1-fga) + fga)
      sl[..., 0:3] = c * 256
      #sl[..., 3] = ca * 256
    work(image, (X, Y), S, 0)
    work(image, (X, Y), S, 1)
