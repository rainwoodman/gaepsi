from warnings import warn

warn("ccode module is deprecated")

from gaepsi._gaepsiccode import pmin
from gaepsi._gaepsiccode import k0, kline, koverlap, akline, akoverlap
from gaepsi._gaepsiccode import peanohilbert
from gaepsi._gaepsiccode import camera
from render import color
from image import rasterize

