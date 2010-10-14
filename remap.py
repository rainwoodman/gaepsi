from numpy.linalg import qr
from numpy import matrix, array, ones_like
from numpy import mgrid
from numpy import diag
from numpy import sign
from numpy import dot,inner
from matplotlib.pyplot import *
from numpy import float64,int32
from numpy import sum
from numpy import bitwise_or
from numpy.linalg import det
from numpy import zeros, ones, arange
from numpy import ceil, floor
from numpy import newaxis
def AABB(E, O = None): 
  "construct the AABB of a box given by E a list of edge row vectors"
  D = E.shape[0]
  E = array(E)
  if D == 2:
    list = array([zeros(D), E[0,:], E[1,:], E[0,:] + E[1,:] ])
  if D == 3:
    list = array([zeros(D), E[0,:], E[1,:], E[2,:], 
                  E[0,:] + E[1,:], 
                  E[0,:] + E[2,:],
                  E[1,:] + E[2,:], 
                  E[0,:] + E[1,:] + E[2,:]])
  min = list.min(axis = 0)
  max = list.max(axis = 0)
  if O != None:
    min += O
    max += O
  return min,max

def cutmask(TXY, BOX): 
  "returns a mask of points that is not in the bounding BOX"
  D = TXY.shape[1]
  # Be tolerent of numerical errors
  # enlarge the target bounding box by e
  e = ones(D) * 1e-6
  LMASK = TXY < -e
  RMASK = TXY > (BOX + e)
  LMASK = bitwise_or.reduce(LMASK, 1)
  RMASK = bitwise_or.reduce(RMASK, 1)
  MASK = LMASK | RMASK
  return MASK

def remap(M, XY) :
  """ cubic remapping: (ref to arxiv1003.3178v1),
      this is an improved algorithm.
      M is the transformation integer matrix. Column vectors
      XY is the list of source coordinates, normalized to [0,1). Row vectors.

      returns 
       TXY, the transformed coordinates,
       BOX, the bounding box in the transformed frame,
       QT, the inverse of the normalized
       outMASK, the mask of points that failed to shift in(shall be empty)
      
  """
  M = matrix(M)
  D = M.shape[0]
  
  # Schmidt orthogonization,
  # Q is the new orthnormal basis
  # diagonal elements of R gives the
  # zooming
  Q,R = qr(M)

  BOX = abs(diag(R))

  # in the original coord, the new bounding box is E
  E = Q * diag(abs(diag(R)))
  print 'matrix M', M
  print 'unfolded box size', BOX
  QT = Q.T
  print 'unity check QT * Q'
  print QT * Q
  print 'singular check q'
  print det(Q)

  # XY is not a numpy matrix
  # inner in numpy reduces the last index of both matrix,
  # therefore we use Q.T the transpose of Q.
  # TXY = XY * Q (TXY and XY are row vectors, thus we use Q on the right)
  TXY = inner(XY, QT)

  # calculate the bounding box,
  # and the min/max cell vector
  # notice that AABB takes a list of vectors,
  # but E is a list of column vectors.
  min, max = AABB(E.T)
  IMAX = int32(ceil(max))
  IMIN = int32(floor(min))
  print min, max
  print IMIN, IMAX

  # Find the suitable cells that intersects the bounding box
  if D == 2:
    r = mgrid[IMIN[0]:IMAX[0]+1, IMIN[1]:IMAX[1]+1]
  if D == 3:
    r = mgrid[IMIN[0]:IMAX[0]+1, IMIN[1]:IMAX[1]+1, IMIN[2]:IMAX[2]+1]
  r.shape = (r.shape[0], -1)
  r = r.T
  # now we start shifting the points into the bounding box
  # outMASK == the points not in the box
  outMASK = cutmask(TXY,BOX)
  print 'trying', r.shape[0], 'cells'
  n = 0
  for I in r:
    n = n + 1
    O = inner(QT,  I)
    # a matrix is a collection of column vectors, 
    # AABB takes row vectors, thus the transposing.
    min, max = AABB(QT.A.T, O)
    # if the cell doesn't intersect the box, skip it
    if (max < 0).any() or (min > BOX).any():
      continue
    # only shfit the arrays that are not in the box
    tmp = TXY[outMASK,:] + O
    # newMASK: points still not in the box
    newMASK = cutmask(tmp, BOX)
    # notnewMASK: points newly shifted into the box
    notnewMASK = ~ newMASK
    # prepare a mask for the points newly shifted into the box
    # with respect to the entire set of points
    newinMASK = outMASK.copy()
    newinMASK[outMASK] = notnewMASK
    # save the newly shifted-in points
    TXY[newinMASK,:] = tmp[notnewMASK]
    # update the mask of out-of-box points
    outMASK[outMASK] = newMASK
    print I, n, '/', r.shape[0]
    if not newMASK.any(): 
      break;
  print "finished."
  print newMASK.sum(), outMASK.sum()

  return TXY,QT,BOX,outMASK

"""
N = 100
XY = float64(mgrid[0:N,0:N,0:N])/N
XY.shape=(3, -1)
XY = XY.T
#M = matrix([[4,4,3],[3,4,-4],[1,1,1]]).T
#M = matrix([[1,1,1],[3,0,1],[1,0,0]]).T
M = matrix([[7,6,6],[6,-3,2],[-1,4,1]]).T
TXY,QT,BOX,MASK = remap(M, XY)
"""

def paint3(TXY=None, BOX=None, QT=None, MASK=None) :
  if TXY!=None:
    TX = TXY[:,0]
    TY = TXY[:,1]
    TZ = TXY[:,2]
  subplot(211)
  if TXY!=None:
    plot(TX, TY, '. ', alpha=1)
  if QT!=None:
    arrow(0,0, QT[0,0], QT[1,0], label='e1')
    arrow(0,0, QT[0,1], QT[1,1], label='e2')
    arrow(0,0, QT[0,2], QT[1,2], label='e3')
  if BOX!=None:
    fill( [0,0,BOX[0], BOX[0]], [0,BOX[1],BOX[1],0], color='black', alpha=0.2)

  axis('equal')
  subplot(212)
  if TXY!=None:
    plot(TX, TZ, ', ', alpha=0.5)
  if QT!=None:
    arrow(0,0, QT[0,0], QT[2,0], label='e1')
    arrow(0,0, QT[0,1], QT[2,1], label='e2')
    arrow(0,0, QT[0,2], QT[2,2], label='e3')
  if BOX!=None:
    fill( [0,0,BOX[0], BOX[0]], [0,BOX[2],BOX[2],0], color='black', alpha=0.2)
  axis('equal')
  draw()


  #if raw_input('exit to quit the loop') == 'exit': break;
def paint2(TXY=None, BOX=None, QT=None, MASK=None) :
  if TXY!=None:
    TX = TXY[:,0]
    TY = TXY[:,1]
  if TXY!=None:
    plot(TX, TY, '. ', alpha=1)
  if QT!=None:
    arrow(0,0, QT[0,0], QT[1,0], label='e1')
    arrow(0,0, QT[0,1], QT[1,1], label='e2')
  if BOX!=None:
    fill( [0,0,BOX[0], BOX[0]], [0,BOX[1],BOX[1],0], color='black', alpha=0.2)

  axis('equal')
  draw()
  
