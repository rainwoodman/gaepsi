from numpy import eye
from numpy import asarray
from numpy.linalg import norm
from numpy import cross, dot, inner
from numpy import zeros
from numpy import tan
from numpy import divide
from numpy import newaxis
from numpy import sum

class Camera(object):
  def __init__(self, pos, near, far, fov, target, up):
    self.near = near
    self.far = far
    self.fov = fov
    self.target = target
    self.up = up
    self.pos = pos

  def map(self, positions, viewport):
    """ maps 3D position to 2D dev position. viewport size is (w, h), returns the 2D coordinate and distance (x, y, d). d is to the camera in unspecified units,  when d < 0 the point is not in the camera's near and far range"""
    c = Context()
    c.lookat(target=self.target, pos=self.pos, up=self.up)
    c.perspective(near= self.near, far=self.far, fov=self.fov / 360. * 3.1415, aspect = 1.0 * viewport[0]/ viewport[1])
    c.viewport(viewport[0], viewport[1])
    distance = zeros(positions.shape[0])
    devpos = c.transform(positions, distance=distance)
    devpos[:, 0:2] = (devpos[:, 0:2] + 1.0 ) * 0.5 * viewport
    badz = (devpos[:, 2] < -1.0) | (devpos[:, 2] > 1.0)
    devpos[:, 2] = distance
    devpos[badz, 2] = -1
    return devpos

class Context:
  def __init__(self):
    self.identity()

  def lookat(self, pos, target, up):
    pos = asarray(pos)
    self.pos = pos
    target = asarray(target)
    up = asarray(up)

    dir = target - pos
    dir /= norm(dir)
    side = cross(dir, up)
    side /= norm(side)
    up = cross(side, dir)
    up /= norm(up)
    
    m1 = zeros((4,4))
    m1[0, 0:3] = side
    m1[1, 0:3] = up
    m1[2, 0:3] = -dir
    m1[3, 3] = 1
    
    tran = eye(4)
    tran[0:3, 3] = -pos
    m2 = dot(m1, tran)

    self.eye = m2

  def perspective(self, near, far, fov, aspect=1.0):
    persp = zeros((4,4))
    persp[0, 0] = 1.0 / tan(fov) / aspect
    persp[1, 1] = 1.0 / tan(fov)
    persp[2, 2] = - (1. *(far + near)) / (far - near)
    persp[2, 3] = - (2. * far * near) / (far - near)
    persp[3, 2] = -1
    persp[3, 3] = 0
    self.camera = persp
    print persp

  def ortho(self, l, r, b, t, near, far):
    ortho = zeros((4,4))
    ortho[0, 0] = 2.0 / (r - l)
    ortho[1, 1] = 2.0 / (t - b)
    ortho[2, 2] = -2.0 / (far - near)
    ortho[3, 3] = 1
    ortho[0, 3] = - (r + l) / (r - l)
    ortho[1, 3] = - (t + b) / (t - b)
    ortho[2, 3] = - (f + n) / (f - n)
    self.camera = ortho

  def identity(self):
    self.camera = eye(4)
    self.eye = eye(4)

  def viewport(self, w, h):
    self.view = asarray([w, h])

  def transform(self, pos, out=None, distance=None):
    """ transform pos to viewport positions, if distance is not None, put distance from the camera there """
    matrix = dot(self.camera, self.eye)
    homo = zeros((pos.shape[0], 4))
    homo[:, 0:3] = pos[:, 0:3]
    homo[:, 3] = 1.0
    int = dot(homo, matrix.T)
    if out is None: out = zeros((pos.shape[0], 3))
    divide(int[:, 0:3], int[:, 3, newaxis], out[:, 0:3])
    if distance is not None:
      distance[:] = sum((pos - self.pos) **2, axis=1) ** 0.5
    return out

GL = Context()
