#cython: embedsignature=True
#cython: cdivision=True
import numpy
cimport numpy

cdef inline void DieseFunktionFrustum(numpy.ndarray frustum, numpy.ndarray matrix):
    """by -=sinuswutz=- from glTerrian"""
    frustum[0,:] = matrix[3,:] - matrix[0,:] #rechte plane berechnen
    frustum[1,:] = matrix[3,:] + matrix[0,:] #linke plane berechnen
    frustum[2,:] = matrix[3,:] + matrix[1,:] #unten plane berechnen
    frustum[3,:] = matrix[3,:] - matrix[1,:] #oben plane berechnen
    frustum[4,:] = matrix[3,:] - matrix[2,:] #ferne plane berechnen
    frustum[5,:] = matrix[3,:] + matrix[2,:] #nahe plane berechnen
    frustum[...] /= ((frustum[:,:-1] ** 2).sum(axis=-1)**0.5)[:, None]
  
cdef inline int DieseFunktion(double frustum[6][3], double AABB[8][3]) nogil:
    """ Diese Funktion liefert 
        0 zurück, wenn die geprüften coordinaten nicht sichtbar sind,
        1 zurück, wenn die coords teilweise sichtbar sind und 
        2 zurück, wenn alle coords sichtbar sind  
        by -=sinuswutz=- from glTerrian
    """
    cdef int cnt=0, vor=0, i, j
    #cnt : zählt, bei wie vielen ebenen alle punkte davor liegen, 
    #vor: zählt für jede ebene, wieviele punkte davor liegen
    for i in range(6):  #für alle ebenen...
      vor = 0 
      for j in range(8):   #für alle punkte...
        if AABB[j][0] * frustum[i][0] \
         + AABB[j][1] * frustum[i][1] \
         + AABB[j][2] * frustum[i][2] \
         + frustum[i][3] > 0: 
          vor = vor + 1

      # alle ecken hinter der ebene, ist nicht sichtbar!
      if vor == 0: return 0 
      # alle vor der ebene, merken und weitermachen    
      if vor == 8: cnt = cnt + 1 

    #liegt komplett im frustum
    if cnt == 6: return 2  
  
    # liegt teilweise im frustum;
    return 1


cdef inline int LiangBarskyClip(double num, double denom, double * tE, double * tL) nogil:
   cdef double t
   if denom == 0: return num <= 0.0
   t = num / denom
   if denom > 0:
     if t > tL[0]: return 0
     if t > tE[0]: tE[0] = t
   else:
     if t < tL[0]: tL[0] = t
     if t < tE[0]: return 0
   
cdef inline int LiangBarsky(double pos[3], double size[3], double p0[3], double dir[3], double * tL, double * tE) nogil:
   """ LiangBarksy test for line from p0 towards dir with box corner at 'pos' of size 'size',
       returns 1 if intersects, and set tL, tE to the lower and upper bound of t.
       otherwise returns 0 and put junk in tL and tE.
   """
   for d in range(3):
     if not LiangBarskyClip(pos[d] - p0[d], dir[d], tE, tL): return 0
     if not LiangBarskyClip(p0[d] - (pos[d] + size[d]), - dir[d], tE, tL): return 0
   return 1
