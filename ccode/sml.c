#include "defines.h"

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

#define max(a, b) ((a) > (b)?(a):(b))
#define min(a, b) ((a) < (b)?(a):(b))

struct rtree_t {
	struct rtree_node_t * pool;
	intptr_t used;
	size_t size;
};

struct rtree_node_t {
/* if type == leaf, first_child = first par, if type == nonleaf first_child = first child node*/
	int type;  /*0 = leaf, 1= nonleaf, nonlastsibling, 2=lastsibling */
/* number of children or pars */
	int length;
/* bounding box */
	float bot[3];
	float top[3];
	intptr_t parent;
	intptr_t first_child;
};


static void rtree_build(struct rtree_t * rtree, PyArrayObject * pos, size_t npar) {
	/* build a rtree, in input, pos is sorted by the key */

	size_t leaves = (npar + 11)/ 12;
	size_t nonleaves = 0;
	size_t t;
	for(t = (leaves + 7)/ 8; t >= 1; t = (t + 7)/ 8) {
		nonleaves += t;
		if(t == 1) break;
	}

	rtree->size = leaves + nonleaves;
	rtree->pool = malloc(sizeof(struct rtree_node_t) * rtree->size);
	rtree->used = 0;
	intptr_t i;
	/* fill the leaves*/
	for(i = 0; i < leaves; i++) {
		struct rtree_node_t * node = &rtree->pool[i];
		node->first_child = i * 12;
		node->type = 0;
		node->length = min(npar - i * 12, 12);
		int d;
		for(d = 0; d < 3; d++) {
			node->bot[d] = *(float*)PyArray_GETPTR2(pos, node->first_child, d);
			node->top[d] = *(float*)PyArray_GETPTR2(pos, node->first_child, d);
		}
		intptr_t j;
		for(j = 0; j < node->length; j++) {
			for(d = 0; d < 3; d++) {
				node->bot[d] = fmin(node->bot[d], *(float*)PyArray_GETPTR2(pos, node->first_child + j,d));
				node->top[d] = fmax(node->top[d], *(float*)PyArray_GETPTR2(pos, node->first_child + j,d));
			}
		}
		rtree->used ++;
	}
	intptr_t child_cur = 0;
	intptr_t parent_cur = leaves;
	for(t = (leaves + 7)/ 8; t >= 1; t = (t + 7)/ 8) {
		for(i = 0; i < t; i++) {
			struct rtree_node_t * node = &rtree->pool[parent_cur + i];
			node->type = 1;
			node->first_child = child_cur + i * 8;
			node->length = min(parent_cur - child_cur - i * 8, 8);
			intptr_t j;
			int d;
			struct rtree_node_t * child = &rtree->pool[node->first_child];
			for(d = 0; d < 3; d++) {
				node->bot[d] = child[0].bot[d];
				node->top[d] = child[0].top[d];
			}
			for(j = 0; j < node->length; j++) {
				child[j].parent = parent_cur + i;
				for(d = 0; d < 3; d++) {
					node->bot[d] = fmin(node->bot[d], child[j].bot[d]);
					node->top[d] = fmax(node->top[d], child[j].top[d]);
				}
			}
			child[j-1].type += 2;
		rtree->used ++;
		}
		child_cur = parent_cur;
		parent_cur += t;
		if(t == 1) break;
	}
}

static int intersect(float top1[], float bot1[], float top2[], float bot2[]) {
	int d;
/*
	printf("%g %g %g - %g %g %g x %g %g %g - %g %g %g",
		top1[0], top1[1], top1[2],
		bot1[0], bot1[1], bot1[2],
		top2[0], top2[1], top2[2],
		bot2[0], bot2[1], bot2[2]);
*/
	for(d = 0 ;d < 3; d++) {
		if(bot1[d] > top2[d]) {
/*
			printf("false\n");
*/
			return 0;
		}
		if(bot2[d] > top1[d]) {
/*
			printf("false\n");
*/
			return 0;
		}
	}
//	printf("true\n");
	return 1;
}
static void add_to_list(intptr_t ipar, PyArrayObject * pos, float p[3], intptr_t neighbours[], float dist[], int n, int * nused) {
	int d;
	float d2 = 0.0;
	for(d = 0; d < 3; d++) {
		float s = *(float*)PyArray_GETPTR2(pos, ipar, d) - p[d];
		d2 += s * s;
	}
	float dd = sqrt(d2);
	if(*nused == n && dd > dist[n - 1]) return;
	int i;

	int ip = 0;
	while(ip < *nused && dist[ip] < dd) ip++;

	for(i = min(*nused, n - 1); i > ip; i--) {
		dist[i] = dist[i - 1];
		neighbours[i] = neighbours[i - 1];
	}
	dist[ip] = dd;
	neighbours[ip] = ipar;
	if(*nused < n) (*nused)++;
}
static int rtree_neighbours(struct rtree_t * rtree, float p[3], PyArrayObject * pos, size_t npar,
		intptr_t neighbours[], float dist[], int n, int *nused, float * hhint) {
	float hh = 1.0;
	if(hhint == NULL) { hhint = &hh; }
	if(n > npar) n = npar;
	float bot[3];
	float top[3];
	int d;
	struct rtree_node_t * N = rtree->pool;
	int rt = 0;
	do {
		for(d = 0; d < 3; d++) {
			bot[d] = p[d] - *hhint;
			top[d] = p[d] + *hhint;
		}
		intptr_t cur = rtree->used - 1;
		*nused = 0;
		rt = 0;
		while(1) {
			if(intersect(N[cur].top, N[cur].bot, top, bot)) {
			rt ++;
				if(N[cur].type == 1 || N[cur].type == 3) {
					cur = N[cur].first_child;
					continue;
				}
				if(N[cur].type == 0 || N[cur].type == 2) {
					int j;
					for(j = 0; j < N[cur].length; j++) {
						add_to_list(N[cur].first_child + j, pos, p, neighbours, dist, n, nused);
					}
				}
			}
			if(cur == rtree->used - 1) break;
			if(N[cur].type == 0 || N[cur].type == 1) {
				cur++;
				continue;
			}
			while((N[cur].type == 2 || N[cur].type == 3)) {
				cur = N[cur].parent;
				if(cur == rtree->used - 1) break;
			}
			if(cur == rtree->used - 1) break;
			cur++;
		}
		(*hhint) *= 2.0;
	} while(*nused < n);
	return rt;
}

typedef struct  {
	PyArrayObject * locations;
	PyArrayObject * mass;
	npy_intp npar;
} PrivateData;

static double mass_est(float * dist, float * mass, int n, double h0) {
	double rho_sph = 0;
	npy_intp j;
	for(j = 0; j < n; j++) {
		float ma = mass[j];
		if(h0 == 0.0) {
			if(dist[j] == 0.0) {
				rho_sph += ma * k0f(0.0); /* / (h0 * h0 * h0) is canceled with multiplication*/
			} else continue;
		} else {
			rho_sph += ma * k0f(dist[j] / h0); /* / (h0 * h0 * h0) is canceled with multiplication*/
		}
//		printf("h0 = %g, rho_sph = %g, ma = %g, dist=%g\n", h0, rho_sph, ma, p->dist);
	}
	return 4 * 3.14 / 3.0 * rho_sph;
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
	PrivateData m;
	if(! PyArg_ParseTupleAndKeywords(args, kwds, "O!O!i", kwlist,
		&PyArray_Type, &locations, 
		&PyArray_Type, &mass, 
		&NGB)) return NULL;

	struct rtree_t t = {0};

	m.locations = (PyArrayObject*) PyArray_Cast(locations, NPY_FLOAT);
	m.mass = (PyArrayObject*) PyArray_Cast(mass, NPY_FLOAT);
	m.npar = PyArray_DIM(locations, 0);
	npy_intp dims[] = {m.npar};
	PyArrayObject * sml = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_FLOAT);

	double msum = 0.0;
	npy_intp i;
	for(i = 0; i < m.npar; i++) {
		msum += *((float*)PyArray_GETPTR1(m.mass, i));
	}
	double mmean = msum / m.npar;
	rtree_build(&t, m.locations, m.npar);
    #pragma omp parallel for private(i) schedule(dynamic, 1000)
	for(i = 0; i < m.npar; i++) {
		int d;
		int n;
		float pos[3];
		int center[3];
		for(d = 0; d < 3; d++) {
			pos[d] = *((float*)PyArray_GETPTR2(m.locations, i, d));
		}
		float dist[256];
		intptr_t nei[256];
		float mass[256];
		float max_dist = 0;
		float min_dist = 0;
		int nused = 0;
		int j;
		float hhint = 1.0;
		int r = rtree_neighbours(&t, pos, m.locations, m.npar, nei, dist, 256, &nused, &hhint);
		for(j = 0; j < nused; j++) {
			mass[j] = *((float*)PyArray_GETPTR1(m.mass, nei[j]));
		}
		min_dist = dist[0];
		max_dist = dist[nused - 1];

		/* iterate to find a converged density & sml, using rho_j =W(h_j) */
		double m_expt = NGB * mmean;

		/* mass_est is monotonic increasing with h0 */
		/* m0 = mass_est(0), m2 = mass_est(inf) */
		double h0 = 0;
		double h2 = 10 * max_dist;
		double m2 = mass_est(dist, mass, nused, h2);
		double m0 = mass_est(dist, mass, nused, h0);
		double h1;
		double m1;
		if(m0 > m_expt) {
			h1 = h0; 
		} else if(m2 < m_expt) {
			h1 = h2;
		} else {
			/*now the solution is within [h0, h1] */ 
			h1 = 0.5 * (h0 + h2);
			m1 = mass_est(dist, mass, nused, h1);
			//printf("m = %g %g %g %g, h= %g %g %g\n", m0, m1, m2, m_expt, h0, h1, h2);
			while(fabs(h0 - h2) / h1 > 1e-3) {
				if((m1 - m_expt) * (m0 - m_expt) >0) {
					h0 = h1;
					m0 = m1;
				} else {
					h2 = h1;
					m2 = m1;
				}
				h1 = 0.5 * (h0 + h2);
				m1 = mass_est(dist, mass, nused, h1);
				//printf("m = %g %g %g %g, h= %g %g %g\n", m0, m1, m2, m_expt, h0, h1, h2);
			}
		}
		h1 = fmax(h1, min_dist);
		/* use a max smoothing length of the nearest 32th neighbour distance */
		h1 = fmin(h1, max_dist);
		//printf("%ld %g %g %d\n", i, h1, hhint, r);
		*((float*)PyArray_GETPTR1(sml, i)) = h1;
	}
	Py_DECREF(m.locations);
	Py_DECREF(m.mass);
	return (PyObject*)sml;
}

static PyMethodDef module_methods[] = {
	{"sml", (PyCFunction) sml, METH_KEYWORDS, sml_doc_string },
	{NULL}
};
void HIDDEN gadget_initsml(PyObject * m) {
	PyObject * sml_f = PyCFunction_New(module_methods, NULL);
	PyModule_AddObject(m, "sml", sml_f);
}
