from matplotlib import is_string_like
from numpy import array, zeros_like
from numpy import linspace, logspace
from numpy import meshgrid
from scipy.weave import inline

from gadget.kernel import kernel_box_values 
from gadget.kernel import kernel_box_bins
from gadget.kernel import kernel_box_deta
import time

def image(field, npixels=[10,10], xrange=None, yrange=None, zrange=None) :
  boxsize = field.boxsize
  origin = field.origin
  if xrange == None: xrange = [origin[0], origin[0] + boxsize[0]]
  if yrange == None: yrange = [origin[1], origin[1] + boxsize[1]]
  if zrange == None: zrange = [origin[2], origin[2] + boxsize[2]]

  field.ensure_quadtree()
  quadtree = field.quadtree

  pixelsize = 1.0 * array([xrange[1] - xrange[0], 
              yrange[1] - yrange[0]]) / npixels
  pixelX = (linspace(xrange[0], xrange[1], npixels[0] + 1) + 0.5 * pixelsize[0])[:-1]
  pixelY = (linspace(yrange[0], yrange[1], npixels[1] + 1) + 0.5 * pixelsize[1])[:-1]
  X, Y = meshgrid(pixelX, pixelY)
  image = zeros_like(X)

  pos = field['locations']
  sml = field['sml']
  value = field['default']

  print "pixel size =", pixelsize
  print "pixel volume=", (zrange[1] - zrange[0]) * pixelsize[0] * pixelsize[1]
  print "making image %d x %d" % (npixels[0], npixels[1])

  image_flat = image.ravel()
  X_flat = X.ravel()
  Y_flat = Y.ravel()

  oldkey = 0
  plist = array([])
  keys = {}
  print time.clock()
  for ipixel in range(image.size):
    xpixel = X_flat[ipixel]
    ypixel = Y_flat[ipixel]
    fullplist, key = quadtree.list(xpixel, ypixel)
    if not keys.has_key(key):
      z = pos[fullplist, 2]
      plist = fullplist[ (z > zrange[0]) & (z < zrange[1])]
      keys[key] = plist
    else :
      plist = keys[key]

    image_flat[ipixel] += render_plist(plist, pos, sml, value, xpixel, ypixel, pixelsize)
      
  del keys
  print time.clock()
  return image

def render_plist(plist, pos, sml, value, xpixel, ypixel, pixelsize) :
  ccode = r"""
#line 28 "test.py"
int bins = kernel_box_bins;

double deta = kernel_box_deta;
double deta2 = deta * deta;
double Xps_2 = 0.5 * pixelsize[0];
double Yps_2 = 0.5 * pixelsize[1];
double rt = 0.0;
	for(int i = 0; i < Nplist[0]; i++) {
		int ip = plist[i];
		double z = POS2(ip, 2);
		double x = POS2(ip, 0);
		double y = POS2(ip, 1);

		double addbit = 0.0;
		double inv_sml = 1.0 / sml[ip];

		double Xps_2_sml = Xps_2 * inv_sml;
		double Yps_2_sml = Yps_2 * inv_sml;
		/* pixel center in particle frame measured by sml, 
           range from 0 to 2 */
		double Xpc = ((double)xpixel - x) * inv_sml + 1.0;
		double Ypc = ((double)ypixel - y) * inv_sml + 1.0;
		double Xpl = Xpc - Xps_2_sml;
		double Xpr = Xpc + Xps_2_sml;
		double Ypl = Ypc - Yps_2_sml;
		double Ypr = Ypc + Yps_2_sml;

		/* no overlaps, skip the particle */
		if(Xpl >= 2.0) continue;
		if(Ypl >= 2.0) continue;
		if(Xpr < 0.0) continue;
		if(Ypr < 0.0) continue;

		/* pixel larger than particle, 
           crop the pixel */
		if(Xpl < 0.0 ) Xpl = 0.0;
		if(Ypl < 0.0 ) Ypl = 0.0;
		if(Xpr > 2.0) Xpr = 2.0;
		if(Ypr > 2.0) Ypr = 2.0;

		/* particle is within the pixel, 
           add the entire particle in */
		if(Xpl == 0.0 && Ypl == 0.0 && Xpr == 2.0 && Ypr == 2.0) {
			addbit = 1.0;
			rt += addbit * value[ip];
			continue;
		}

		/* find which bin the pixel edges sit in */
		int x0 = Xpl / deta;
		int y0 = Ypl / deta;
		int x1 = Xpr / deta;
		int y1 = Ypr / deta;

		/* possible if Xpr == 2.0 or Ypr == 2.0*/
		if(x1 == bins) x1 = bins - 1;
		if(y1 == bins) y1 = bins - 1;

#define RW(a) ((a) > bins -1)?(bins - 1):(a)
#define LW(a) ((a) < 0)?0:(a)
		double x0y0 = KERNEL_BOX_VALUES4(x0, y0, x0, y0);
		if(x1 == x0 && y0 == y1) {
			addbit += x0y0 * (Ypr - Ypl) * (Xpr - Xpl) / deta2;
			rt += addbit * value[ip];
			continue;
		}
		double x0y1 = KERNEL_BOX_VALUES4(x0, y1, x0, y1);
		double ldy = (y0 + 1) * deta - Ypl;
		double rdy = Ypr - y1 * deta;
		if(x1 == x0 && y1 == y0 + 1) {
			addbit += x0y0 * ldy * (Xpr - Xpl) / deta2;
			addbit += x0y1 * rdy * (Xpr - Xpl) / deta2;
			rt += addbit * value[ip];
			continue;
		}
		double x1y0 = KERNEL_BOX_VALUES4(x1, y0, x1, y0);
		double ldx = (x0 + 1) * deta - Xpl;
		double rdx = Xpr - x1 * deta;
		if(x1 == x0 + 1 && y1 == y0) {
			addbit += x0y0 * ldx * (Ypr - Ypl) / deta2;
			addbit += x1y0 * rdx * (Ypr - Ypl) / deta2;
			rt += addbit * value[ip];
			continue;
		}
		double x1y1 = KERNEL_BOX_VALUES4(x1, y1, x1, y1);
		if(x1 == x0 + 1 && y1 == y0 + 1) {
			addbit += x0y0 * ldx * ldy / deta2;
			addbit += x1y0 * rdx * ldy / deta2;
			addbit += x0y1 * ldx * rdy / deta2;
			addbit += x1y1 * rdx * rdy / deta2;
			rt += addbit * value[ip];
			continue;
		}
		double left = KERNEL_BOX_VALUES4(x0, RW(y0 + 1), x0, LW(y1 - 1));
		if(x1 == x0 && y1 > y0 + 1) {
			addbit += x0y0 * ldy * (Xpr - Xpl) / deta2;
			addbit += x0y1 * rdy * (Xpr - Xpl) / deta2;
			addbit += left * (Xpr - Xpl) / deta;
			rt += addbit * value[ip];
			continue;
		}
		double top = KERNEL_BOX_VALUES4(RW(x0 + 1), y0, LW(x1 - 1), y0);
		if(x1 > x0 + 1 && y1 == y0) {
			addbit += x0y0 * ldx * (Ypr - Ypl) / deta2;
			addbit += x1y0 * rdx * (Ypr - Ypl) / deta2;
			addbit += top * (Ypr - Ypl) / deta;
			rt += addbit * value[ip];
			continue;
		}
		double right = KERNEL_BOX_VALUES4(x1, RW(y0 + 1), x1, LW(y1 - 1));
		if(x1 == x0 + 1 && y1 > y0 + 1) {
			addbit += x0y0 * ldx * ldy / deta2;
			addbit += x1y0 * rdx * ldy / deta2;
			addbit += x0y1 * ldx * rdy / deta2;
			addbit += x1y1 * rdx * rdy / deta2;
			addbit += left * ldx / deta;
			addbit += right * rdx / deta;
			rt += addbit * value[ip];
			continue;
		}
		double bottom = KERNEL_BOX_VALUES4(RW(x0 + 1), y1, LW(x1 - 1), y1);
		if(x1 > x0 + 1 && y1 == y0 + 1) {
			addbit += x0y0 * ldx * ldy / deta2;
			addbit += x1y0 * rdx * ldy / deta2;
			addbit += x0y1 * ldx * rdy / deta2;
			addbit += x1y1 * rdx * rdy / deta2;
			addbit += top * ldy / deta;
			addbit += bottom * rdy / deta;
			rt += addbit * value[ip];
			continue;
		}
		double center = KERNEL_BOX_VALUES4(RW(x0 + 1), RW(y0 + 1), LW(x1 - 1), RW(y1 - 1));
		if(x1 > x0 + 1 && y1 > y0 + 1) {
			addbit += x0y0 * ldx * ldy / deta2;
			addbit += x1y0 * rdx * ldy / deta2;
			addbit += x0y1 * ldx * rdy / deta2;
			addbit += x1y1 * rdx * rdy / deta2;
			addbit += left * ldx / deta;
			addbit += right * rdx / deta;
			addbit += top * ldy / deta;
			addbit += bottom * rdy / deta;
			addbit += center;
			rt += addbit * value[ip];
			continue;
		}
		printf("unhandled %lf; %lf %lf %lf %lf; %lf %lf %lf %lf\n", center, left, right, top, bottom, x0y0, x0y1, x1y1, x1y0);
	}
	return_val = PyFloat_FromDouble(rt);
  """
  return inline(ccode,
         ['xpixel', 'ypixel',
          'pos', 'plist',
          'pixelsize', 'sml',
          'kernel_box_values',
          'kernel_box_deta', 
          'kernel_box_bins', 
          'value'], 
         extra_compile_args=['-Wno-unused', '-O3']);
      
