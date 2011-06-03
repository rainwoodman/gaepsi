from numpy.linalg import qr
from numpy import matrix, array, ones_like
from numpy import mgrid
from numpy import diag
from numpy import sign
from numpy import dot,inner
from numpy import float64,int32,float32
from numpy import sum
from numpy import bitwise_or
from numpy.linalg import det
from numpy import zeros, ones, arange
from numpy import ceil, floor
from numpy import newaxis

import ccode

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

def remap(M, XY=None) :
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
  QT = Q.T

  """
  print 'matrix M', M
  print 'unfolded box size', BOX
  print 'unity check QT * Q'
  print QT * Q
  print 'singular check q'
  print det(Q)
  """
  if XY == None: return BOX
  # XY is not a numpy matrix
  # inner in numpy reduces the last index of both matrix,
  # therefore we use Q.T the transpose of Q.
  # TXY = XY * Q (TXY and XY are row vectors, thus we use Q on the right)
  TXY = float32(inner(XY, QT))

  # calculate the bounding box,
  # and the min/max cell vector
  # notice that AABB takes a list of vectors,
  # but E is a list of column vectors.
  min, max = AABB(E.T)
  IMAX = int32(ceil(max))
  IMIN = int32(floor(min))

  ccode.remap_shift(POS=TXY, ROWVECTORS = float32(QT.T), BOX=float32(BOX), MIN=int32(IMIN), MAX=int32(IMAX));
  return TXY,BOX

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
  
