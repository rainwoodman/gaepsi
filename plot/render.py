from numpy import fromfile
from numpy import zeros, dtype
from numpy import uint8, array, linspace
from numpy import int32, float32
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

def circle(target, X, Y, R, scale=1.0, min=None, max=None, logscale=False, color=[0,0,0]): 
  if len(target.shape) != 3:
     raise ValueError("has to be a rgb bitmap! expecting shape=(#,#,3)")
  image = target
  X = int32(X)
  Y = int32(Y)
  if logscale: R = log10(R)
  if min == None: min = R.min()
  if max == None: max = R.max()
  print min, max
  R = int32((R - min) * scale / (max - min))
  print R.min(), R.max()
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
  inline(ccode, ['X', 'Y', 'image', 'color', 'R'])

def render(array, min, max, logscale=True, levels = [0, 0.2, 0.4, 0.6, 1.0], rmap =[0, 0.5, 1.0, 1.0, 0.2], gmap=[0, 0.0, 0.5, 1.0, 0.2], bmap=[0.0, 0.0, 0.0, 0.3, 1.0], target=None):

  if target != None:
    image = target
  else :
    image = zeros(array.shape, dtype=dtype(('u1', 3)))
  image_flat = image.reshape((-1, 3))
  array_flat = array.ravel()
  bins = 1024 * 16
  ri = uint8(icmap(levels, rmap, bins) * 255)
  gi = uint8(icmap(levels, gmap, bins) * 255)
  bi = uint8(icmap(levels, bmap, bins) * 255)
  print 'start rendering'
  ccode = r"""
int i, j, _bins = bins; 
int log = logscale;
float _min = min;
float _max = max;
for(i = 0; i < Nimage_flat[0]; i++) {
    float v = ARRAY_FLAT1(i);
	if(log) {
		if ( v <= 0.0 ) 
			v = -30;
		else 
			v = log10f(v);
	}
	float value = (v - _min) / (_max - _min);
	int index = value * _bins;
	if(index < 0) index = 0;
	if(index >= _bins) index = _bins - 1;
	IMAGE_FLAT2(i,0) = RI1(index);
	IMAGE_FLAT2(i,1) = GI1(index);
	IMAGE_FLAT2(i,2) = BI1(index);
}
  """
  inline(ccode, ["image_flat", "array_flat", "min", "max", 'rmap', 'gmap', 'bmap', 'ri', 'gi', 'bi', 'bins', 'logscale'])
  return image

