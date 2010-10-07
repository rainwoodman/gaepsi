from matplotlib import is_string_like
from numpy import array, zeros_like
from numpy import linspace, logspace
from numpy import meshgrid
from scipy.weave import inline

from gadget.kernel import kernel_box_values 
from gadget.kernel import kernel_box_bins
from gadget.kernel import kernel_box_deta

def image(field, npixels=[10,10], xrange=None, yrange=None, zrange=None) :
  boxsize = field.boxsize
  if xrange == None: xrange = [0, boxsize]
  if yrange == None: yrange = [0, boxsize]
  if zrange == None: zrange = [0, boxsize]

  field.ensure_hoc()
  hoc = field.hocindex

  ccode = r"""
#line 28 "test.py"
int bins = kernel_box_bins;

double deta = kernel_box_deta;
double deta2 = deta * deta;
double zmin = zrange[0];
double zmax = zrange[1];
/*
printf("%d %d %lf %lf %d\n", i, j, (double)x, (double)y, Nparticles[0]);
*/
int nx = npixels[0];
int ny = npixels[1];
printf("%d %d\n", nx, ny);
for(int i = 0; i < nx; i++) 
for(int j = 0; j < ny; j++) {
	double rt = 0.0;
	double x = pixelx[i];
	double y = pixely[j];
    int xcc = x / (double) hoccellsize;
    int ycc = y / (double) hoccellsize;
    while(xcc < 0) xcc += hocncells;
    while(ycc < 0) ycc += hocncells;
    while(xcc >= hocncells) xcc -= hocncells;
    while(ycc >= hocncells) ycc -= hocncells;
	long long l = xcc * hocncells + ycc;

	PyObject * key = PyInt_FromLong(l);
	PyArrayObject * particles = (PyArrayObject *) PyDict_GetItem(py_hochoc, key);
	Py_DECREF(key);
	for(int pi = 0; pi < particles->dimensions[0]; pi++) {
		int p = *((int *)(particles->data + pi * particles->strides[0]));
		double zp = POS2(p, 2);
		double xp = POS2(p, 0);
		double yp = POS2(p, 1);
		if(zp < zmin || zp > zmax) {
			/*  printf("skipped one particle\n");*/
			continue;
		}
		double addbit = 0.0;
		double psml = sml[p];
		double psx_2 = 0.5 * pixelsize[0] / psml;
		double psy_2 = 0.5 * pixelsize[1] / psml;
		/* range from 0 to 2 */
		double pxp = ((double)x - xp) / psml + 1.0;
		double pyp = ((double)y - yp) / psml + 1.0;
		double lpxp = pxp - psx_2;
		double rpxp = pxp + psx_2;
		double lpyp = pyp - psy_2;
		double rpyp = pyp + psy_2;
		/* no contributions if the pixel do not overlap the kernel */
		/*
		   printf("xy %lf %lf\n", (double)xp[p], (double)yp[p]);
		   if(xp > 1.0 && xp < 1.3)
		   printf("cr: %lf %lf %lf %lf\n", lpxp, lpyp, rpxp, rpyp);
		 */
		if(lpxp >= 2.0) continue;
		if(lpyp >= 2.0) continue;
		if(rpxp < 0.0) continue;
		if(rpyp < 0.0) continue;

		/* crop the pixel */
		if(lpxp < 0.0 ) lpxp = 0;
		if(lpyp < 0.0 ) lpyp = 0;
		if(rpxp > 2.0) rpxp = 2.0;
		if(rpyp > 2.0) rpyp = 2.0;

		if(lpxp == 0.0 && lpyp == 0.0 && rpxp == 2.0 && rpyp == 2.0) {
            /* if the particle is entirely within the pixel, put everything in */
			addbit = 1.0;
			rt += addbit * value[p];
			continue;
		}

		/* find which bin the pixel edges sit in */
		int x0 = lpxp / deta;
		int y0 = lpyp / deta;
		int x1 = rpxp / deta;
		int y1 = rpyp / deta;
		/* possible if rpxp == 2.0 or rpyp == 2.0*/
		if(x1 == bins) x1 = bins - 1;
		if(y1 == bins) y1 = bins - 1;

		/*
		   if(xp > 1.0 && xp < 1.3)
		   printf("re %d %d %d %d\n", lpxp, lpyp, rpxp, rpyp, x0, y0, x1, y1);
		 */
#define RW(a) ((a) > bins -1)?(bins - 1):(a)
#define LW(a) ((a) < 0)?0:(a)
#define PADDBIT ;
		/*
		   if(xp > 1.0 && xp < 1.3) \
		   printf("addbit = %lf\n", addbit);
		 */
		double x0y0 = KERNEL_BOX_VALUES4(x0, y0, x0, y0);
		if(x1 == x0 && y0 == y1) {
			addbit += x0y0 * (rpyp - lpyp) * (rpxp - lpxp) / deta2;
			rt += addbit * value[p];
			PADDBIT
				continue;
		}
		double x0y1 = KERNEL_BOX_VALUES4(x0, y1, x0, y1);
		double ldy = (y0 + 1) * deta - lpyp;
		double rdy = rpyp - y1 * deta;
		if(x1 == x0 && y1 == y0 + 1) {
			addbit += x0y0 * ldy * (rpxp - lpxp) / deta2;
			addbit += x0y1 * rdy * (rpxp - lpxp) / deta2;
			rt += addbit * value[p];
			PADDBIT
				continue;
		}
		double x1y0 = KERNEL_BOX_VALUES4(x1, y0, x1, y0);
		double ldx = (x0 + 1) * deta - lpxp;
		double rdx = rpxp - x1 * deta;
		if(x1 == x0 + 1 && y1 == y0) {
			addbit += x0y0 * ldx * (rpyp - lpyp) / deta2;
			addbit += x1y0 * rdx * (rpyp - lpyp) / deta2;
			rt += addbit * value[p];
			PADDBIT
				continue;
		}
		double x1y1 = KERNEL_BOX_VALUES4(x1, y1, x1, y1);
		if(x1 == x0 + 1 && y1 == y0 + 1) {
			addbit += x0y0 * ldx * ldy / deta2;
			addbit += x1y0 * rdx * ldy / deta2;
			addbit += x0y1 * ldx * rdy / deta2;
			addbit += x1y1 * rdx * rdy / deta2;
			rt += addbit * value[p];
			PADDBIT
				continue;
		}
		double left = KERNEL_BOX_VALUES4(x0, RW(y0 + 1), x0, LW(y1 - 1));
		if(x1 == x0 && y1 > y0 + 1) {
			addbit += x0y0 * ldy * (rpxp - lpxp) / deta2;
			addbit += x0y1 * rdy * (rpxp - lpxp) / deta2;
			addbit += left * (rpxp - lpxp) / deta;
			rt += addbit * value[p];
			PADDBIT
				continue;
		}
		double top = KERNEL_BOX_VALUES4(RW(x0 + 1), y0, LW(x1 - 1), y0);
		if(x1 > x0 + 1 && y1 == y0) {
			addbit += x0y0 * ldx * (rpyp - lpyp) / deta2;
			addbit += x1y0 * rdx * (rpyp - lpyp) / deta2;
			addbit += top * (rpyp - lpyp) / deta;
			rt += addbit * value[p];
			PADDBIT
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
			rt += addbit * value[p];
			PADDBIT
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
			rt += addbit * value[p];
			PADDBIT
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
			rt += addbit * value[p];
			PADDBIT
				continue;
		}
		printf("unhandled %lf; %lf %lf %lf %lf; %lf %lf %lf %lf\n", center, left, right, top, bottom, x0y0, x0y1, x1y1, x1y0);
	}
	IMAGE2(i,j) = rt;
}
  """


  pixelsize = 1.0 * array([xrange[1] - xrange[0], yrange[1] - yrange[0]]) / npixels
  pixelx = (linspace(xrange[0], xrange[1], npixels[0] + 1) + 0.5 * pixelsize[0])[:-1]
  pixely = (linspace(yrange[0], yrange[1], npixels[1] + 1) + 0.5 * pixelsize[1])[:-1]
  X, Y = meshgrid(pixelx, pixely)
  image = zeros_like(X)

  pos = field['locations']
  sml = field['sml']
  value = field['default']

  print "pixel size =", pixelsize
  print "pixel volume=", (zrange[1] - zrange[0]) * pixelsize[0] * pixelsize[1]
  print "making image %d x %d" % (npixels[0], npixels[1])

  hochoc = hoc.hoc
  hoccellsize = hoc.cellsize
  hocncells = hoc.ncells
  inline(ccode,
         ['pixelx', 'pixely',
          'pos',
          'pixelsize', 'sml',
          'npixels',
          'hochoc',
          'hoccellsize',
          'hocncells',
          'kernel_box_values',
          'kernel_box_deta', 
          'kernel_box_bins', 
          'value', 'zrange',
          'image'], 
         extra_compile_args=['-Wno-unused', '-O3']);
      
  return image
