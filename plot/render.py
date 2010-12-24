from numpy import fromfile
from numpy import zeros, dtype, ones
from numpy import uint8, array, linspace
from numpy import int32, float32, int16
from scipy.weave import inline
from numpy import log10
def icmap(levels, cmap, bins):
  cmap = array(cmap)
  detail = zeros(bins)
  for i in range(len(cmap) -1):
    start = int(levels[i] * (bins - 1))
    end = int(levels[i + 1] * (bins -1))
    detail[start:end] = linspace(cmap[i], cmap[i+1], end - start, endpoint=False)
  detail[-1] = cmap[-1]
  return detail

class Colormap:
  def __init__(self, levels, r=None, g=None, b=None, h=None, s=None, v=None, a=None):
    if r != None:
      self.set_rgb(levels, r, g, b, a)
    else :
      self.set_hsv(levels, h, s, v, a)
  def set_rgb(self, levels, r, g, b, a = None):
    bins = 1024 * 16
    ri = icmap(levels, r, bins)
    gi = icmap(levels, g, bins)
    bi = icmap(levels, b, bins)
    self.ri = ri
    self.gi = gi
    self.bi = bi
    if a != None:
      ai = icmap(levels, a, bins)
    else:
      ai = ones(bins)
    self.ai = ai

  def set_hsv(self, levels, h, s, v, a = None):
    """ h is 0 to 360, s is 0 to 1, v is 0 to 1"""
    bins = 1024 * 16
    hi = icmap(levels, h, bins)
    si = icmap(levels, s, bins)
    vi = icmap(levels, v, bins)
    ri = zeros(bins)
    gi = zeros(bins)
    bi = zeros(bins)
    f = hi % 60
    l = uint8(hi / 60)
    l = l % 6
    p = vi * (1 - si)
    q = vi * (1 - si * f / 60)
    t = vi * (1 - si * (60 - f) / 60)
    ri[l==0] = vi[l==0]
    gi[l==0] = t[l==0]
    bi[l==0] = p[l==0]

    ri[l==1] = q[l==1]
    gi[l==1] = vi[l==1]
    bi[l==1] = p[l==1]

    ri[l==2] = p[l==2]
    gi[l==2] = vi[l==2]
    bi[l==2] = t[l==2]
  
    ri[l==3] = p[l==3]
    gi[l==3] = q[l==3]
    bi[l==3] = vi[l==3]

    ri[l==4] = t[l==4]
    gi[l==4] = p[l==4]
    bi[l==4] = vi[l==4]

    ri[l==5] = vi[l==5]
    gi[l==5] = p[l==5]
    bi[l==5] = q[l==5]
    self.ri = ri
    self.gi = gi
    self.bi = bi
    if a != None:
      ai = icmap(levels, a, bins)
    else:
      ai = ones(bins)
    self.ai = ai

def circle(target, X, Y, R, scale=1.0, min=None, max=None, logscale=False, color=[0,0,0]): 
  if len(target.shape) != 3:
     raise ValueError("has to be a rgb bitmap! expecting shape=(#,#,3)")
  image = target
  X = int32(X)
  Y = int32(Y)
  if logscale: R = log10(R)
  if min == None: min = R.min()
  if max == None: max = R.max()
  if min == max: 
    min = min - 0.5
    max = max + 0.5
  print 'internal min, max = ', R.min(), R.max()

  R = int32((R - min) * scale / (max - min))
  print 'pixel min, max =', R.min(), R.max()
  R[R < 0] = 0
  R[R > scale ] = scale
  color = float32(color)
  ccode = r"""
int i;
unsigned char r = (float)color[0] * 255;
unsigned char g = (float)color[1] * 255;
unsigned char b = (float)color[2] * 255;
printf("%d\n", NR[0]);
printf("%d %d %d\n", Nimage[0], Nimage[1], Nimage[2]);
for(i = 0; i < NR[0]; i++) {
#define SET(x, y) { \
	if(x>=0 && x<Nimage[0] && y>=0 && y<Nimage[1]) {\
		IMAGE3(x,y,0) = r; \
		IMAGE3(x,y,1) = g; \
		IMAGE3(x,y,2) = b; \
	} \
	}
	int radius = R[i];
	int cx = X[i];
	int cy = Y[i];
	int error = - radius;
	int x = radius;
	int y = 0;
	if(radius == 0) {
		SET(cx, cy);
	}
	while(x >= y) {
		SET(cx + x, cy + y);
		if(x) SET(cx - x, cy + y);
		if(y) SET(cx + x, cy - y);
		if(x && y) SET(cx - x, cy - y);
		if(x != y) {
			SET(cx + y, cy + x);
			if(y) SET(cx - y, cy + x);
			if(x) SET(cx + y, cy - x);
			if(x && y) SET(cx - y, cy - x);
		}
		error += y;
		++y;
		error += y;
		if(error >= 0) {
			--x;
			error -= x;
			error -= x;
		}
	}
}
"""
  inline(ccode, ['X', 'Y', 'image', 'color', 'R', 'scale'])

def render(array, min, max, logscale=True, colormap=None, target=None):
  if colormap == None:
    colormap = Colormap(levels = [0, 0.2, 0.4, 0.6, 1.0], 
                         r=[0, 0.5, 1.0, 1.0, 0.2], 
                         g=[0, 0.0, 0.5, 1.0, 0.2], 
                         b=[0.0, 0.0, 0.0, 0.3, 1.0])

  if target != None:
    rgb = target
  else :
    rgb = zeros(array.shape, dtype=dtype(('u1', 3)))

  rgb_flat = rgb.reshape((-1, 3))
  array_flat = array.ravel()

  ri = uint8(colormap.ri * 255)
  gi = uint8(colormap.gi * 255)
  bi = uint8(colormap.bi * 255)
  ai = uint8(colormap.ai * 100)
  bins = len(ri)
  print 'start rendering'
  ccode = r"""
int i, j, _bins = bins; 
int log = logscale;
float _min = min;
float _max = max;
int skipped = 0;
for(i = 0; i < Narray_flat[0]; i++) {
    float v = ARRAY_FLAT1(i);
	if(log) {
		if ( v <= 0.0) {
			skipped ++;
			continue;
		}
		v = log10(v);
	}
	float value = (v - _min) / (_max - _min);
	int index = value * _bins;
	if(index < 0) index = 0;
	if(index >= _bins) index = _bins - 1;
	int r,g,b,a;
	r = RGB_FLAT2(i, 0);
	g = RGB_FLAT2(i, 1);
	b = RGB_FLAT2(i, 2);
	a = AI1(index);
	int at = 100 - a;
	r = (r * at + a * RI1(index)) / 100;
	g = (g * at + a * GI1(index)) / 100;
	b = (b * at + a * BI1(index)) / 100;
	if(r > 255) r = 255;
	if(g > 255) g = 255;
	if(b > 255) b = 255;
	RGB_FLAT2(i,0) = r;
	RGB_FLAT2(i,1) = g;
	RGB_FLAT2(i,2) = b;
}
printf("skipped %d pixels\n", skipped);
  """
  inline(ccode, ["rgb_flat", "array_flat", "min", "max", 'ri', 'gi', 'bi', 'ai', 'bins', 'logscale'])
  return rgb

