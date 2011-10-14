#include <Python.h>
#include <numpy/arrayobject.h>
#include <time.h>
#include "defines.h"
#define HIDDEN __attribute__ ((visibility ("hidden")))

#define sml_doc_string \
"keywords: locations, N=32, returns the smls"

extern float k0f(const float eta);

unsigned int icbrt64(unsigned long long x) {
  int s;
  unsigned int y;
  unsigned long long b;

  y = 0;
  for (s = 63; s >= 0; s -= 3) {
    y += y;
    b = 3*y*((unsigned long long) y + 1) + 1;
    if ((x >> s) >= b) {
      x -= b << s;
      y++;
    }
  }
  return y;
}

typedef struct {
	float min[3];
	float max[3];
	float cellsize[3];
	int ncellx;
	PyArrayObject * locations;
	PyArrayObject * mass;
	int * ccount;
	intptr_t * link;
	intptr_t * cell;
	size_t npar;
} MeshT;

typedef struct _NgbT{
	float dist2;
	intptr_t ipar;
	struct _NgbT * next;
} NgbT;

NgbT * ngbt_insert_sorted(NgbT * list, NgbT * node) {
	if(list == NULL) {
		node->next = NULL;
		return node;
	}
	/* greater than head, then node becomes the new head */
	if(node->dist2 > list->dist2) {
		node->next = list;
		return node;
	}
	NgbT * p, * q;
	for(p = list; p != NULL; p = q) {
		q = p->next;
		if(q) {
			if(q->dist2 < node->dist2) {
				p->next = node;
				node->next = q;
				break;
			}
		} else {
			/* on the tail*/
			p->next = node;
			node->next = NULL;
			/* will terminate anyways*/
			break;
		}
	}
	return list;
}
intptr_t par2cell(MeshT * m, intptr_t ipar, int * cellid_out) {
	float pos[3];
	int d;
	int cellid_internal[3];
	int * cellid;
	if(cellid_out != NULL) {
		cellid = cellid_out;
	} else {
		cellid = cellid_internal;
	}
	intptr_t cellindex = 0;
	for(d = 0; d < 3; d++) {
		pos[d] = *((float*)PyArray_GETPTR2(m->locations, ipar, d));
		cellid[d] = (pos[d] - m->min[d]) / m->cellsize[d];
		if(cellid[d] >= m->ncellx) cellid[d] = m->ncellx - 1;
		if(cellid[d] < 0) fprintf(stderr, "cellid < 0 shall never happen\n");
		cellindex = cellindex * m->ncellx + cellid[d];
	}
	return cellindex;
}
static PyObject * sml(PyObject * self, 
	PyObject * args, PyObject * kwds) {
	static char * kwlist[] = {
		"locations", "mass", "N",
		NULL
	};
	int NGB = 32;
	PyArrayObject * locations = NULL;
	PyArrayObject * mass = NULL;
	MeshT m;
	memset(&m, 0, sizeof(m));
	if(! PyArg_ParseTupleAndKeywords(args, kwds, "O!O!i", kwlist,
		&PyArray_Type, &locations, 
		&PyArray_Type, &mass, 
		&NGB)) return NULL;
	m.locations = (PyArrayObject*) PyArray_Cast(locations, NPY_FLOAT);
	m.mass = (PyArrayObject*) PyArray_Cast(mass, NPY_FLOAT);
	m.npar = PyArray_DIM(locations, 0);
	m.ncellx = icbrt64(m.npar/8); /* 8 particles per cell*/
	npy_intp dims[] = {m.npar};
	PyArrayObject * sml = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_FLOAT);
	m.link = PyMem_New(intptr_t, m.npar);
	m.cell = PyMem_New(intptr_t, m.ncellx * m.ncellx * m.ncellx);
	m.ccount = PyMem_New(int, m.ncellx * m.ncellx * m.ncellx);

	memset(m.cell, -1, sizeof(intptr_t) * m.ncellx * m.ncellx * m.ncellx);
	memset(m.ccount, 0, sizeof(int) * m.ncellx * m.ncellx * m.ncellx);
	memset(m.link, -1, sizeof(intptr_t) * m.npar);

	intptr_t i;
	/* calculate the cell size */
	for(i = 0; i < m.npar; i++) {
		int d;
		float pos[3];
		for(d = 0; d < 3; d++) {
			pos[d] = *((float*)PyArray_GETPTR2(m.locations, i, d));
		}
		if(i == 0) {
			for(d = 0; d < 3; d++) {
				m.min[d] = pos[d];
				m.max[d] = pos[d];
			}
			continue;
		}
		for(d = 0; d < 3; d++) {
			if(m.min[d] > pos[d]) m.min[d] = pos[d];
			if(m.max[d] < pos[d]) m.max[d] = pos[d];
		}
	}
	int d;
	for(d = 0; d < 3; d++) {
		m.cellsize[d] = (m.max[d] - m.min[d]) / m.ncellx;
	}
	/* populate the cells */
	for(i = 0; i < m.npar; i++) {
		intptr_t cellindex = par2cell(&m, i, NULL);
		/* insert the particle to the link list of the cell*/
		intptr_t head = m.cell[cellindex];
		m.cell[cellindex] = i;
		m.link[i] = head;
		m.ccount[cellindex]++;
	}

	float maxdist = (m.cellsize[0] + m.cellsize[1] + m.cellsize[2]) * m.ncellx;
    #pragma omp parallel private(i) 
    {
		/* calculate the sml */
		NgbT * ngb_pool = malloc(sizeof(NgbT) * NGB);
		#pragma omp for schedule(dynamic, 1000)
		for(i = 0; i < m.npar; i++) {
			int d;
			int n;
			float pos[3];
			int center[3];
			for(d = 0; d < 3; d++) {
				pos[d] = *((float*)PyArray_GETPTR2(m.locations, i, d));
			}
			par2cell(&m, i, center);
			size_t count = 0; /* total number of particles looked */
			int r;
			NgbT * ngb_head = NULL;
			ngb_head = NULL;
			/* reset the ngb array */
			int ngb_pool_used = 0;
			for(n = 0; n < NGB; n++) {
				ngb_pool[n].dist2 = maxdist * maxdist;
				ngb_pool[n].next = NULL;
			}
			for(r = 0; r <= m.ncellx; r++) {
				int c[3];
				int bot[3];
				int top[3];
				for(d = 0; d < 3; d++) {
					bot[d] = center[d] - r;
					if(bot[d] < 0) bot[d] += m.ncellx;
					top[d] = center[d] + r;
					if(top[d] >= m.ncellx) top[d] -= m.ncellx ;
				}
				int c0[] = { bot[0], top[0]};
				int c1[] = { bot[1], top[1]};
				int c2[] = { bot[2], top[2]};
				int i0, i1, i2;
				count = 0;
				int added = 0;
				for(i0 = 0; i0 < 2; i0+=r?1:2)
				for(i1 = 0; i1 < 2; i1+=r?1:2)
				for(i2 = 0; i2 < 2; i2+=r?1:2) {
					c[0] = c0[i0];
					c[1] = c1[i1];
					c[2] = c2[i2];
					intptr_t cellindex = 0;
					for(d = 0; d < 3; d++) {
						cellindex = cellindex * m.ncellx + c[d];
					}
					count += m.ccount[cellindex];
					intptr_t next = m.cell[cellindex];
					while(next != (intptr_t)-1) {
						float dist2 = 0.0;
						for(d = 0; d < 3; d++) {
							float s = pos[d] - *((float*)PyArray_GETPTR2(m.locations, next, d));
							dist2 += s * s;
						}
						if(ngb_pool_used < NGB) {
							/* if havn't got enough neighbours */
							ngb_pool[ngb_pool_used].dist2 = dist2;
							ngb_pool[ngb_pool_used].ipar = next;
							/* so that ngb_head is the furthest */
							ngb_head = ngbt_insert_sorted(ngb_head, &ngb_pool[ngb_pool_used]);
							ngb_pool_used ++;
							added = 1;
						} else {
							/* if nearer than the head */
							if(dist2 < ngb_head->dist2) {
								/* pop out the head and reinsert */
								ngb_head->dist2 = dist2;
								ngb_head->ipar = next;
								ngb_head = ngbt_insert_sorted(ngb_head->next, ngb_head);
								added = 1;
							}
						}
						next = m.link[next];
					}
				}
				float r5 = r > 0?r-1:r;
				if(ngb_head->dist2 < r5 * r5 * m.cellsize[0] * m.cellsize[0]
				 &&ngb_head->dist2 < r5 * r5 * m.cellsize[1] * m.cellsize[1]
				 &&ngb_head->dist2 < r5 * r5 * m.cellsize[2] * m.cellsize[2]) break;
			}
			/* iterate to find a converged density & sml, using rho_j =W(h_j) */
			double h0 = sqrt(ngb_head->dist2);
			double h1 = 0;
			double rho_sph;
			double mass_sph = 0.0;
			NgbT * p = NULL;
			for(p = ngb_head; p != NULL; p = p->next) {
				mass_sph += *((float*)PyArray_GETPTR1(m.mass, p->ipar));
			}
			int icount = 0;
			while(icount < 128) {
				rho_sph = 0;
				for(p = ngb_head; p != NULL; p = p->next) {
					float ma = *((float*)PyArray_GETPTR1(m.mass, p->ipar));
					rho_sph += ma * k0f(sqrt(p->dist2) / h0) / (h0 * h0 * h0);
				}
				h1 = cbrt(mass_sph / ( 4 * 3.14 / 3 * rho_sph));
				if(fabs(h1 - h0) / h0 < 1e-2) break;
				h0 = h1;
				icount++;
			}
			if(icount >=128) {
//				printf("warning a sml failed to converge\n");
				h1 = 10 *sqrt(ngb_head->dist2);
			}
			*((float*)PyArray_GETPTR1(sml, i)) = h1;

	/*
			printf("mass=%g, icount = %d, h1 =%f h0 = %f head = %f, rho = %g\n",
			*(float*)(PyArray_GETPTR1(m.mass, i)),
			icount, h1, h0, sqrt(ngb_head->dist2), rho_sph
			);
			printf("%f %f %f %d %d ", pos[0], pos[1], pos[2], count, r);
			for(ngb_head; ngb_head != NULL; ngb_head = ngb_head->next) {
				printf("%f ", sqrt(ngb_head->dist2));
			}
			printf("\n");
	*/
		}
		free(ngb_pool);
	}
	PyMem_Del(m.link);
	PyMem_Del(m.cell);
	Py_DECREF(m.locations);
	Py_DECREF(m.mass);
	return (PyObject*)sml;
}

static PyMethodDef module_methods[] = {
	{"sml", (PyCFunction) sml, METH_KEYWORDS, sml_doc_string },
	{NULL}
};
void HIDDEN gadget_initsml(PyObject * m) {
	import_array();
	PyObject * sml_f = PyCFunction_New(module_methods, NULL);
	PyModule_AddObject(m, "sml", sml_f);
}
