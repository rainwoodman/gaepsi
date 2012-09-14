#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
typedef __int128_t zorder_t;
//typedef int64_t ipos_t;
#define BITS 40
/*
 * g = numpy.ogrid.__getitem__([slice(0, 2) * 8)
 * g.reverse()
 * 
 * utab = reduce(numpy.add, [ (1<<(3*i)) * b for i, b in enumerate(g)]).ravel()
 *
 * */
static ipos_t utab[256] = {
    0x000000, 0x000001, 0x000008, 0x000009,
    0x000040, 0x000041, 0x000048, 0x000049,
    0x000200, 0x000201, 0x000208, 0x000209,
    0x000240, 0x000241, 0x000248, 0x000249,
    0x001000, 0x001001, 0x001008, 0x001009,
    0x001040, 0x001041, 0x001048, 0x001049,
    0x001200, 0x001201, 0x001208, 0x001209,
    0x001240, 0x001241, 0x001248, 0x001249,
    0x008000, 0x008001, 0x008008, 0x008009,
    0x008040, 0x008041, 0x008048, 0x008049,
    0x008200, 0x008201, 0x008208, 0x008209,
    0x008240, 0x008241, 0x008248, 0x008249,
    0x009000, 0x009001, 0x009008, 0x009009,
    0x009040, 0x009041, 0x009048, 0x009049,
    0x009200, 0x009201, 0x009208, 0x009209,
    0x009240, 0x009241, 0x009248, 0x009249,
    0x040000, 0x040001, 0x040008, 0x040009,
    0x040040, 0x040041, 0x040048, 0x040049,
    0x040200, 0x040201, 0x040208, 0x040209,
    0x040240, 0x040241, 0x040248, 0x040249,
    0x041000, 0x041001, 0x041008, 0x041009,
    0x041040, 0x041041, 0x041048, 0x041049,
    0x041200, 0x041201, 0x041208, 0x041209,
    0x041240, 0x041241, 0x041248, 0x041249,
    0x048000, 0x048001, 0x048008, 0x048009,
    0x048040, 0x048041, 0x048048, 0x048049,
    0x048200, 0x048201, 0x048208, 0x048209,
    0x048240, 0x048241, 0x048248, 0x048249,
    0x049000, 0x049001, 0x049008, 0x049009,
    0x049040, 0x049041, 0x049048, 0x049049,
    0x049200, 0x049201, 0x049208, 0x049209,
    0x049240, 0x049241, 0x049248, 0x049249,
    0x200000, 0x200001, 0x200008, 0x200009,
    0x200040, 0x200041, 0x200048, 0x200049,
    0x200200, 0x200201, 0x200208, 0x200209,
    0x200240, 0x200241, 0x200248, 0x200249,
    0x201000, 0x201001, 0x201008, 0x201009,
    0x201040, 0x201041, 0x201048, 0x201049,
    0x201200, 0x201201, 0x201208, 0x201209,
    0x201240, 0x201241, 0x201248, 0x201249,
    0x208000, 0x208001, 0x208008, 0x208009,
    0x208040, 0x208041, 0x208048, 0x208049,
    0x208200, 0x208201, 0x208208, 0x208209,
    0x208240, 0x208241, 0x208248, 0x208249,
    0x209000, 0x209001, 0x209008, 0x209009,
    0x209040, 0x209041, 0x209048, 0x209049,
    0x209200, 0x209201, 0x209208, 0x209209,
    0x209240, 0x209241, 0x209248, 0x209249,
    0x240000, 0x240001, 0x240008, 0x240009,
    0x240040, 0x240041, 0x240048, 0x240049,
    0x240200, 0x240201, 0x240208, 0x240209,
    0x240240, 0x240241, 0x240248, 0x240249,
    0x241000, 0x241001, 0x241008, 0x241009,
    0x241040, 0x241041, 0x241048, 0x241049,
    0x241200, 0x241201, 0x241208, 0x241209,
    0x241240, 0x241241, 0x241248, 0x241249,
    0x248000, 0x248001, 0x248008, 0x248009,
    0x248040, 0x248041, 0x248048, 0x248049,
    0x248200, 0x248201, 0x248208, 0x248209,
    0x248240, 0x248241, 0x248248, 0x248249,
    0x249000, 0x249001, 0x249008, 0x249009,
    0x249040, 0x249041, 0x249048, 0x249049,
    0x249200, 0x249201, 0x249208, 0x249209,
    0x249240, 0x249241, 0x249248, 0x249249,
};

/*
g = numpy.ogrid.__getitem__([slice(0, 2)] * 8)
g.reverse()
ctab = reduce(numpy.add, [ (1<<((i % 3) * 3 + i/3)) * b for i, b in enumerate(g)]).ravel()
ctab = ['0x%02x, ' % c for c in ctab]
for i in range(0, len(ctab), 4):
  print ctab[i], ctab[i+1], ctab[i+2], ctab[i+3]

 * */
static uint8_t ctab[256] = {
0x00,  0x01,  0x08,  0x09, 
0x40,  0x41,  0x48,  0x49, 
0x02,  0x03,  0x0a,  0x0b, 
0x42,  0x43,  0x4a,  0x4b, 
0x10,  0x11,  0x18,  0x19, 
0x50,  0x51,  0x58,  0x59, 
0x12,  0x13,  0x1a,  0x1b, 
0x52,  0x53,  0x5a,  0x5b, 
0x80,  0x81,  0x88,  0x89, 
0xc0,  0xc1,  0xc8,  0xc9, 
0x82,  0x83,  0x8a,  0x8b, 
0xc2,  0xc3,  0xca,  0xcb, 
0x90,  0x91,  0x98,  0x99, 
0xd0,  0xd1,  0xd8,  0xd9, 
0x92,  0x93,  0x9a,  0x9b, 
0xd2,  0xd3,  0xda,  0xdb, 
0x04,  0x05,  0x0c,  0x0d, 
0x44,  0x45,  0x4c,  0x4d, 
0x06,  0x07,  0x0e,  0x0f, 
0x46,  0x47,  0x4e,  0x4f, 
0x14,  0x15,  0x1c,  0x1d, 
0x54,  0x55,  0x5c,  0x5d, 
0x16,  0x17,  0x1e,  0x1f, 
0x56,  0x57,  0x5e,  0x5f, 
0x84,  0x85,  0x8c,  0x8d, 
0xc4,  0xc5,  0xcc,  0xcd, 
0x86,  0x87,  0x8e,  0x8f, 
0xc6,  0xc7,  0xce,  0xcf, 
0x94,  0x95,  0x9c,  0x9d, 
0xd4,  0xd5,  0xdc,  0xdd, 
0x96,  0x97,  0x9e,  0x9f, 
0xd6,  0xd7,  0xde,  0xdf, 
0x20,  0x21,  0x28,  0x29, 
0x60,  0x61,  0x68,  0x69, 
0x22,  0x23,  0x2a,  0x2b, 
0x62,  0x63,  0x6a,  0x6b, 
0x30,  0x31,  0x38,  0x39, 
0x70,  0x71,  0x78,  0x79, 
0x32,  0x33,  0x3a,  0x3b, 
0x72,  0x73,  0x7a,  0x7b, 
0xa0,  0xa1,  0xa8,  0xa9, 
0xe0,  0xe1,  0xe8,  0xe9, 
0xa2,  0xa3,  0xaa,  0xab, 
0xe2,  0xe3,  0xea,  0xeb, 
0xb0,  0xb1,  0xb8,  0xb9, 
0xf0,  0xf1,  0xf8,  0xf9, 
0xb2,  0xb3,  0xba,  0xbb, 
0xf2,  0xf3,  0xfa,  0xfb, 
0x24,  0x25,  0x2c,  0x2d, 
0x64,  0x65,  0x6c,  0x6d, 
0x26,  0x27,  0x2e,  0x2f, 
0x66,  0x67,  0x6e,  0x6f, 
0x34,  0x35,  0x3c,  0x3d, 
0x74,  0x75,  0x7c,  0x7d, 
0x36,  0x37,  0x3e,  0x3f, 
0x76,  0x77,  0x7e,  0x7f, 
0xa4,  0xa5,  0xac,  0xad, 
0xe4,  0xe5,  0xec,  0xed, 
0xa6,  0xa7,  0xae,  0xaf, 
0xe6,  0xe7,  0xee,  0xef, 
0xb4,  0xb5,  0xbc,  0xbd, 
0xf4,  0xf5,  0xfc,  0xfd, 
0xb6,  0xb7,  0xbe,  0xbf, 
0xf6,  0xf7,  0xfe,  0xff, 
};

static inline zorder_t _truncate(zorder_t key, int order) {
  return (key >> (order * 3)) << (order * 3);
}

static inline void _flatten(zorder_t key, ipos_t *x, ipos_t * y) {
  *x = 0; *y = 0;
  int base = 0;
  while(key >0) {
    *x += (key & 0x7) << base;
    key >>= 3;
    *y += (key & 0x7) << base;
    key >>= 3;
    base += 3;
  }
}

static inline zorder_t _xyz2ind (ipos_t x, ipos_t y, ipos_t z) {
    zorder_t ind = 0;
    x &= (1L << BITS) - 1;
    y &= (1L << BITS) - 1;
    z &= (1L << BITS) - 1;
    int i;
    for (i = 0; i < BITS; i+= 8) {
      ind |= (zorder_t) utab[(uint8_t) x] << (i*3+0);
      ind |= (zorder_t) utab[(uint8_t) y] << (i*3+1);
      ind |= (zorder_t) utab[(uint8_t) z] << (i*3+2);
      x >>= 8;
      y >>= 8;
      z >>= 8;
    }
    if( ind < 0) {
          abort();
    }
    return ind;
}

zorder_t masks[] = {
     ((zorder_t)0111111111111111111111L << 63) | ((zorder_t)0111111111111111111111L),
     ((zorder_t)0222222222222222222222L << 63) | ((zorder_t)0222222222222222222222L),
     ((zorder_t)0444444444444444444444L << 63) | ((zorder_t)0444444444444444444444L),
};

static ipos_t inline _ind2x(zorder_t ind) {
    zorder_t comp = ind & masks[0];
    ipos_t x = 0;
    uint8_t raw;

    int i;
    for( i = 0; i < BITS; i+=8) {
      raw = comp;
      comp >>= 8;
      raw |= comp;
      comp >>= 8;
      raw |= comp;
      comp >>= 8;
      x |= ((ipos_t)(ctab[raw]) << i);
      if(!comp) return x;

    }
    return x;
}

void inline _ind2xyz(zorder_t ind, ipos_t * x, ipos_t * y, ipos_t * z) {
  *x = _ind2x(ind);
  *y = _ind2x(ind>>1);
  *z = _ind2x(ind>>2);
}

static int _boxtest(zorder_t key, int order, zorder_t point) {
  return 0 == ((key ^ point) >> (order * 3));
}

void inline _diff(zorder_t ind1, zorder_t ind2, ipos_t d[3]) {
  d[0] = _ind2x(ind2);
  d[1] = _ind2x(ind2>>1);
  d[2] = _ind2x(ind2>>2);
  d[0] -= _ind2x(ind1);
  d[1] -= _ind2x(ind1>>1);
  d[2] -= _ind2x(ind1>>2);
}
int inline _diff_order(zorder_t ind1, zorder_t ind2) {
  zorder_t diff = ind2 ^ ind1;
  int r = 0;
  if (diff >> 96) { diff >>= 96; r += 32; }
  if (diff >> 48) { diff >>= 48; r += 16; }
  if (diff >> 24) { diff >>= 24; r += 8; }
  if (diff >> 12) { diff >>= 12; r += 4; }
  if (diff >> 6) { diff >>= 6; r += 2; }
  if (diff >> 3) { diff >>= 3; r += 1; }
  if (diff) {r += 1;}
  return r;
}

static char * _format_key(zorder_t key) {
  static char buf[100];
  ipos_t ix, iy, iz;
  _ind2xyz(key, &ix, &iy, &iz);

  sprintf(buf, "%lx %lx, %ld %ld %ld", key, ix, iy, iz);
  return strdup(buf);
}
static int _AABBtest(zorder_t key, int order, zorder_t AABB[2]) {
  /**** returns -2 if key, order fully in AABB, returns -1 if partially in,
 *      0 if not in  ***/
  int d, i;
  zorder_t l, x0, x1, r;
  int in = 0;
  zorder_t m = ((zorder_t)1 << (3*order)) - 1;
  for(d=0; d<3; d++) {
    l = (key & masks[d]) & ~m;
    r = l + (masks[d] & m);
    x0 = AABB[0] & masks[d];
    x1 = AABB[1] & masks[d];
    /* r <= x0 because it's open on the right */
  //  printf ("x0 %d x1 %d l %d, r %d\n", _ind2x(x0 >> d), _ind2x(x1 >> d),
   //    _ind2x(l >> d), _ind2x(r >>d));
    if (x1 < x0) abort();
    if (l > x1 || r < x0) return 0;
    /* r <= x1 because both are open on the right */
    if (l > x0 && r < x1) in++;
  }
  if(in == 3) return -2;
  return -1;
}

#if 0

int main() {
    ipos_t ix;
    ipos_t iy;
    ipos_t iz;
    int xstep=1, ystep=1, zstep=1;
    int count = 0;
    int i, j;
    for(ix = 0; ix < (1 << 30) && xstep > 0; ix += xstep, xstep <<=1) {
    for(iy = 0; iy < (1 << 30) && ystep > 0; iy += ystep, ystep <<=1) {
    for(iz = 0; iz < (1 << 30) && zstep > 0; iz += zstep, zstep <<=1) {
        zorder_t ind = _xyz2ind(ix, iy, iz);
        ipos_t x, y, z;
        _ind2xyz(ind, &x, &y, &z);
        if(x != ix || y != iy || z != iz) {
            abort();
        }
        count++;
    }
    }
    }
    printf("xyz ind done: %d\n", count);
    for(i = 1; i < 30; i++) {
      ix = 1<<i; iy = 1<<i; iz = 1<<i;
      zorder_t ind = _xyz2ind(ix, iy, iz);
      int bit = 1 << (i-1);
      zorder_t ind2 = _xyz2ind(ix + bit, iy + bit, iz + bit);
      if(!_boxtest(ind, i, ind2)) {
        printf("inside fail: %lx %d %lx\n", ind, i, ind2);
        abort();
      }
      if(_boxtest(ind, i - 1, ind2)) {
        printf("outside fail: %lx %d %lx\n", ind, i, ind2);
        abort();
      }
    }
    printf("boxtest done: %d\n", count);

    for(i = 1; i < 30; i++) {
      ix = 1<<i; iy = 1<<i; iz = 1<<i;
      zorder_t AABB[2];
      zorder_t ind = _xyz2ind(ix, iy, iz);
      int bit = 1 << (i-1);
      zorder_t ind2 = _xyz2ind(ix + bit, iy + bit, iz + bit);
      AABB[0] = ind;
      AABB[1] = ind2;
      for (j = 0; j < i; j++) {
        if(!_AABBtest(ind, j, AABB)) {
          printf("inside fail: %lx %lx %lx %lx %lx %lx %d\n", AABB[0], AABB[1], ind, j);
          abort();
        }
        /* the following will for sure fail, need to come up with a better test*/
        if(_AABBtest(ind2, j, AABB)) {
          printf("outside fail: %lx %lx %lx %lx %lx %lx %d\n", AABB[0], AABB[1], ind2, j);
          abort();
        }
      }
    }
    printf("AABBtest done: %d\n", count);

    return 0;
}
#endif

/* from numpy*/
#define SMALL_QUICKSORT 17
#define PYA_QS_STACK 128

#define ZORDER_LT(a, b) ((a) < (b))

static int compare_zorder(zorder_t *v, zorder_t *u, void *NOT_USED) {
    return (*v > *u) - (*v < *u);
}

#define INTP_SWAP(a, b) {intptr_t tmp = (b); (b) = (a); (a) = tmp;}
static int
aquicksort_zorder(zorder_t *v, intptr_t* tosort, intptr_t num, void *NOT_USED)
{
    zorder_t vp;
    intptr_t *pl, *pr;
    intptr_t *stack[PYA_QS_STACK], **sptr=stack, *pm, *pi, *pj, *pk, vi;

    pl = tosort;
    pr = tosort + num - 1;

    for (;;) {
        while ((pr - pl) > SMALL_QUICKSORT) {
            /* quicksort partition */
            pm = pl + ((pr - pl) >> 1);
            if (ZORDER_LT(v[*pm],v[*pl])) INTP_SWAP(*pm,*pl);
            if (ZORDER_LT(v[*pr],v[*pm])) INTP_SWAP(*pr,*pm);
            if (ZORDER_LT(v[*pm],v[*pl])) INTP_SWAP(*pm,*pl);
            vp = v[*pm];
            pi = pl;
            pj = pr - 1;
            INTP_SWAP(*pm,*pj);
            for (;;) {
                do ++pi; while (ZORDER_LT(v[*pi],vp));
                do --pj; while (ZORDER_LT(vp,v[*pj]));
                if (pi >= pj) {
                    break;
                }
                INTP_SWAP(*pi,*pj);
            }
            pk = pr - 1;
            INTP_SWAP(*pi,*pk);
            /* push largest partition on stack */
            if (pi - pl < pr - pi) {
                *sptr++ = pi + 1;
                *sptr++ = pr;
                pr = pi - 1;
            }
            else {
                *sptr++ = pl;
                *sptr++ = pi - 1;
                pl = pi + 1;
            }
        }

        /* insertion sort */
        for (pi = pl + 1; pi <= pr; ++pi) {
            vi = *pi;
            vp = v[vi];
            pj = pi;
            pk = pi - 1;
            while (pj > pl && ZORDER_LT(vp, v[*pk])) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
        if (sptr == stack) {
            break;
        }
        pr = *(--sptr);
        pl = *(--sptr);
    }

    return 0;
}

