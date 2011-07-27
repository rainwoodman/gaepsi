#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>

#define HIDDEN __attribute__ ((visibility ("hidden")))  

typedef struct _Cell {
	intptr_t head_par;
	intptr_t first_child;
	intptr_t parent;
	float bot[3];
	float top[3];
	unsigned int npar;
} Cell;

typedef struct _OctTree {
	PyObject_HEAD
	intptr_t * next;
	Cell * pool;
	Py_ssize_t pool_length;
	Py_ssize_t pool_size;
	PyArrayObject * locations;
	PyArrayObject * sml;
	PyArrayObject * boxsize;
	PyArrayObject * origin;
	Py_ssize_t npar;
	Py_ssize_t skipped;
} OctTree;

typedef struct {
	PyObject_HEAD
	OctTree * tree;
	Py_ssize_t headid;
	Py_ssize_t childid;
	Py_ssize_t parentid;
	PyArrayObject * bottom;
	PyArrayObject * top;
	Py_ssize_t npar;
} OctCell;

static void build_tree(OctTree * tree);
static void add(OctTree * tree, intptr_t ipar, intptr_t icell);
static intptr_t sibling(OctTree * tree, intptr_t icell);
static int inside(OctTree * tree, float pos[3], intptr_t icell);
static int split(OctTree * tree, intptr_t icell);
static intptr_t find(OctTree * tree, float pos[3], intptr_t icell);
static intptr_t parent(OctTree * tree, intptr_t icell);
static int full(OctTree * tree, intptr_t icell);
static void getpos(OctTree * tree, float pos[3], intptr_t ipar);
static float getsml(OctTree * tree, intptr_t ipar);
static int hit(OctTree * tree, const float s[3], const float dir[3], const float dist, const intptr_t icell);
static size_t trace(OctTree * tree, const float s[3], const float dir[3], const float dist, intptr_t ** pars, size_t * size);

static PyTypeObject OctTreeType = {
	PyObject_HEAD_INIT(NULL)
	0, "ccode.OctTree",
	sizeof(OctTree)
};

static PyTypeObject OctCellType = {
	PyObject_HEAD_INIT(NULL)
	0, "ccode.OctCell",
	sizeof(OctCell)
};

static PyObject * OctTree_new(PyTypeObject * type, 
	PyObject * args, PyObject * kwds) {
	OctTree * self;
	self = (OctTree *)type->tp_alloc(type, 0);
	return (PyObject *) self;
}


static void OctTree_dealloc(OctTree * self) {
	Py_XDECREF(self->sml);
	Py_XDECREF(self->locations);
	Py_XDECREF(self->boxsize);
	Py_XDECREF(self->origin);
	PyMem_Del(self->next);
	PyMem_Del(self->pool);
	self->ob_type->tp_free((PyObject*) self);

}

static void OctCell_dealloc(OctCell * self) {
	self->ob_type->tp_free((PyObject*) self);

	Py_DECREF(self->tree);
}


static int OctTree_init(OctTree * self, PyObject * args, PyObject * kwds) {
	PyArrayObject * locations = NULL;
	PyArrayObject * sml = NULL;
	PyArrayObject * boxsize;
	PyArrayObject * origin;
	size_t npar = 0;
	static char * kwlist[] = {"locations", "origin", "boxsize", "sml", NULL};

	self->pool = PyMem_New(Cell, 16);
	self->pool_size = 16;
	self->pool_length = 0;

	if(! PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O!|O!", kwlist,
		&PyArray_Type, &locations,
		&PyArray_Type, &origin, 
		&PyArray_Type, &boxsize, 
		&PyArray_Type, &sml
	)) return -1;
	
	PyMem_Del(self->pool);
	self->pool_size = 0;

    fprintf(stderr, "NDTree_init on %p\n", self);
	locations = (PyArrayObject*) PyArray_Cast(locations, NPY_FLOAT);
	sml = (PyArrayObject*) PyArray_Cast(sml, NPY_FLOAT);
	npar = PyArray_DIM((PyObject*)locations, 0);

    boxsize = (PyArrayObject*) PyArray_Cast(boxsize, NPY_FLOAT);
    origin = (PyArrayObject*) PyArray_Cast(origin, NPY_FLOAT);

	self->sml = sml;
	self->locations = locations;
	self->npar = npar;
	self->origin = origin;
	self->boxsize = boxsize;

	build_tree(self);

	return 0;
}

static PyObject * OctTree_get_cell(OctTree * self, PyObject * args, PyObject * kwds) {
	unsigned long long icell = 0;
	static char * kwlist[] = {"icell", NULL};
	if(! PyArg_ParseTupleAndKeywords(args, kwds, "l", kwlist,
		&icell)) return NULL;
	if(icell > self->pool_length) {
		PyErr_SetString(PyExc_IndexError, "index out of range");
		return NULL;
	}
	OctCell * rt = (OctCell*) OctCellType.tp_alloc(&OctCellType, 0);
	Cell * cell = &self->pool[icell];
	rt->tree = self;
	rt->headid = cell->head_par;
	rt->childid = cell->first_child;
	rt->parentid = cell->parent;
	rt->npar = cell->npar;

	npy_intp dim[] = {3};
	rt->bottom = (PyArrayObject*) PyArray_SimpleNewFromData(1, dim, NPY_FLOAT, cell->bot);
	rt->top = (PyArrayObject*) PyArray_SimpleNewFromData(1, dim, NPY_FLOAT, cell->top);
	Py_INCREF(self);
	
	return (PyObject*)rt;
}

static void free_pars(PyObject * capsule) {
	PyMem_Del(PyCapsule_GetPointer(capsule, NULL));
}

static PyObject * OctTree_trace(OctTree * self, PyObject * args, PyObject * kwds) {
	PyArrayObject * srcpos;
	PyArrayObject * dir;
	double dist;

	static char * kwlist[] = {"src", "dir", "dist", NULL};
	if(! PyArg_ParseTupleAndKeywords(args, kwds, "O!O!d", kwlist,
		&PyArray_Type, &srcpos,
		&PyArray_Type, &dir,
		&dist)) return NULL;

	srcpos = (PyArrayObject*) PyArray_Cast(srcpos, NPY_FLOAT);
	dir = (PyArrayObject*) PyArray_Cast(dir, NPY_FLOAT);
	float s[3];
	float di[3];
	int d;
	for(d = 0; d < 3; d++) {
		s[d] = *((float*)PyArray_GETPTR1(srcpos, d));
		di[d] = *((float*)PyArray_GETPTR1(dir, d));
	}
	intptr_t * pars = NULL;
	size_t size = 0;
	size_t length;
	length = trace(self, s, di, dist, &pars, &size);
	npy_intp dims[] = {length};
	
	PyArrayObject * rt = (PyArrayObject*) PyArray_SimpleNewFromData(1, dims, NPY_INTP, pars);
	PyArray_BASE(rt) = PyCapsule_New(pars, NULL, free_pars);
	Py_DECREF(srcpos);
	Py_DECREF(dir);
	return (PyObject*) rt;
}

static PyObject * OctTree_str(OctTree * self) {
	return PyString_FromFormat("OctTree");
}
static PyObject * OctCell_str(OctCell * self) {
	return PyString_FromFormat("OctCell");
}

static PyMemberDef OctTree_members[] = {
	{"boxsize", T_OBJECT_EX, offsetof(OctTree, boxsize), READONLY,  "boxsize"},
	{"origin", T_OBJECT_EX, offsetof(OctTree, origin), READONLY,  "origin"},
	{"locations", T_OBJECT_EX, offsetof(OctTree, locations), READONLY,  "locations"},
	{"sml", T_OBJECT_EX, offsetof(OctTree, sml), READONLY,  "sml"},
	{"pool_length", T_PYSSIZET, offsetof(OctTree, pool_length), READONLY,  "pool_length"},
	{"pool_size", T_PYSSIZET, offsetof(OctTree, pool_size), READONLY,  "pool_size"},
	{"npar", T_PYSSIZET, offsetof(OctTree, npar), READONLY,  "npar"},
	{"skipped", T_PYSSIZET, offsetof(OctTree, skipped), READONLY,  "skipped"},
	{NULL}
};
static PyMemberDef OctCell_members[] = {
	{"tree", T_OBJECT_EX, offsetof(OctCell, tree), READONLY,  "the tree"},
	{"headid", T_PYSSIZET, offsetof(OctCell, headid), READONLY,  "the tree"},
	{"childid", T_PYSSIZET, offsetof(OctCell, childid), READONLY,  "the tree"},
	{"parentid", T_PYSSIZET, offsetof(OctCell, parentid), READONLY,  "the tree"},
	{"bottom", T_OBJECT_EX, offsetof(OctCell, bottom), READONLY,  "the tree"},
	{"top", T_OBJECT_EX, offsetof(OctCell, top), READONLY,  "the tree"},
	{"npar", T_PYSSIZET, offsetof(OctCell, npar), READONLY,  "the tree"},
	{NULL}
};
static PyMethodDef OctCell_methods[] = {
	{NULL}
};
static PyMethodDef OctTree_methods[] = {
	{"get_cell", (PyCFunction) OctTree_get_cell, 
		METH_KEYWORDS,
		"keywords: i returns a cell\n"
		},
	{"trace", (PyCFunction) OctTree_trace, 
		METH_KEYWORDS,
		"keywords: trace a ray\n"
		},
	{NULL}
};


void HIDDEN gadget_initOctTree (PyObject * m) {

	import_array();
	OctTreeType.tp_dealloc = (destructor) OctTree_dealloc;
	OctTreeType.tp_new = OctTree_new;
	OctTreeType.tp_init = (initproc) OctTree_init;
	OctTreeType.tp_str = (reprfunc) OctTree_str;
	OctTreeType.tp_members = OctTree_members;
	OctTreeType.tp_methods = OctTree_methods;
	OctTreeType.tp_doc = "OctTree(D, pos, sml, origin, boxsize, periodical=True)";
	OctTreeType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;

	OctCellType.tp_dealloc = (destructor) OctCell_dealloc;
	OctCellType.tp_str = (reprfunc) OctCell_str;
	OctCellType.tp_members = OctCell_members;
	OctCellType.tp_methods = OctCell_methods;
	OctCellType.tp_doc = "OctCell";
	OctCellType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;

	if (PyType_Ready(&OctTreeType) < 0) return;
	if (PyType_Ready(&OctCellType) < 0) return;

	Py_INCREF(&OctTreeType);
	Py_INCREF(&OctCellType);
	PyModule_AddObject(m, "OctTree", (PyObject *) &OctTreeType);
	PyModule_AddObject(m, "OctCell", (PyObject *) &OctCellType);
}

static void build_tree(OctTree * tree) {
	tree->next = PyMem_New(intptr_t, tree->npar);
	tree->pool_size = tree->npar / 16;
	if(tree->pool_size <= 16) tree->pool_size = 16;
	tree->pool = PyMem_New(Cell, tree->pool_size);
tryagain:
	tree->pool_length = 0;
	memset(tree->next, -1, sizeof(intptr_t) * tree->npar);
	tree->pool[0].parent = -1;
	int d;
	for(d = 0; d < 3; d++) {
		tree->pool[0].bot[d] = *((float*)PyArray_GETPTR1(tree->origin, d));
		tree->pool[0].top[d] = tree->pool[0].bot[d] + 
							*((float*)PyArray_GETPTR1(tree->boxsize, d));
	}
	tree->pool[0].first_child = -1;
	tree->pool[0].head_par = -1;
	tree->pool[0].npar = 0;
	tree->pool_length ++;

	intptr_t icell = 0;
	intptr_t ipar;
	tree->skipped = 0;
	for(ipar = 0; ipar < tree->npar; ipar++) {
		float pos[3];
		getpos(tree, pos, ipar);
		while(!inside(tree, pos, icell)) {
			icell = parent(tree, icell);
			if(icell == -1) break;
		}
		/* if par not in the root cell, skip it*/
		if(icell == -1) {
			tree->skipped ++;
			continue;
		}

		icell = find(tree, pos, icell);

		while(full(tree, icell)) {
			if(!split(tree, icell)) {
				printf("OCTTREE full %lu mean occupy = %f\n", tree->pool_size, (double)ipar / tree->pool_size);
				size_t trysize = tree->npar / ((double) ipar / tree->pool_size);
				PyMem_Del(tree->pool);
				if(tree->pool_size < trysize && 8 * tree->pool_size > trysize) {
					tree->pool_size = trysize;
				} else tree->pool_size *= 2;
				tree->pool = PyMem_New(Cell , tree->pool_size);
				printf("retry with %lu\n", tree->pool_size);
				goto tryagain;
			}
			icell = find(tree, pos, icell);
		}
		add(tree, ipar, icell);
	}
	if(tree->skipped > 0)
		printf("%ld particles out of cell", tree->skipped);

	/* now recalculate the AABB boxes */
	size_t done = 0;
	char * child_done = malloc(tree->pool_length);
	memset(child_done, 0, tree->pool_length);

	for(icell = 0; icell < tree->pool_length; icell++) {
		if(tree->pool[icell].first_child == -1) {
			child_done[icell] = 8;
		}
	}
	
	while(done < tree->pool_length) {	
		for(icell = 0; icell < tree->pool_length; icell++) {
			if(child_done[icell] > 8) {
				printf("more than 8 children?");
			}
			if(child_done[icell] != 8) continue;
			float * top = tree->pool[icell].top;
			float * bot = tree->pool[icell].bot;
			if(tree->pool[icell].first_child == -1) {
				int first = 1;
				for(ipar = tree->pool[icell].head_par; ipar!= -1; ipar = tree->next[ipar]) {
					float pos[3];
					getpos(tree, pos, ipar);
					float sml = getsml(tree, ipar);
					for(d = 0; d < 3; d++) {
						if(first || pos[d] - sml < bot[d]) bot[d] = pos[d] - sml;
						if(first || pos[d] + sml > top[d]) top[d] = pos[d] + sml;
					}
					first = 0;
				}
			} else {
				int first = 1;
				int i;
				for(i = 0; i < 8 ;i++) {
					intptr_t first_child = tree->pool[icell].first_child;
					float * cbot = tree->pool[first_child + i].bot;
					float * ctop = tree->pool[first_child + i].top;
					for(d = 0; d < 3; d++) {
						if(first || cbot[d] < bot[d]) bot[d] = cbot[d];
						if(first || ctop[d] > top[d]) top[d] = ctop[d];
					}
					first = 0;
				}
			}
			done ++;
			child_done[icell] = 0;
			if(icell != 0) {
				child_done[tree->pool[icell].parent]++;
			}
		}
		printf("updating AABB %lu/%lu done ", done, tree->pool_length);
	}
	free(child_done);
}

static size_t trace(OctTree * tree, const float s[3], const float dir[3], const float dist, intptr_t ** pars, size_t * size) {
	size_t length = 0;
	if(*pars == NULL) {
		*size = 1000;
		*pars = PyMem_New(intptr_t, * size);
	}

	intptr_t icell = 0;
	while(icell != -1) {
		if(hit(tree, s, dir, dist, icell)) {
			if(tree->pool[icell].first_child != -1) {
				icell = tree->pool[icell].first_child;
				continue;
			} else {
				intptr_t ipar;
				for(ipar = tree->pool[icell].head_par;
					ipar != -1;
					ipar = tree->next[ipar]) {
					float pos[3];
					getpos(tree, pos, ipar);
					float sml = getsml(tree, ipar);
					int d ;
					float dist = 0.0;
					float proj = 0.0;
					for(d = 0; d < 3; d++) {
						float dd = pos[d] - s[d];
						proj += dd * dir[d];
						dist += dd * dd;
					}
					if( sml * sml < (dist - proj * proj)) {
						continue;
					}
					if(length == *size) {
						*size *= 2;
						*pars = PyMem_Resize(*pars, intptr_t, *size);
					}
					(*pars)[length] = ipar;
					length ++;
				}
			}
		}
		intptr_t next = -1;
		/* root cell has no parents, the search is end */
		while(icell != 0) {
			/* find the next sibling of parent */
			next = sibling(tree, icell);
			if(next != -1) break;
			/* found a sibling, move there*/
			icell = parent(tree, icell);
		}
		icell = next;
	}
	return length;
}

static int hit(OctTree * tree, const float s[3], const float dir[3], const float dist, const intptr_t icell) {
	extern int pluecker_(const float const dir[3], const float * dist, const float const s2b[3], const float const s2t[3]);
	float * bot = tree->pool[icell].bot;
	float * top = tree->pool[icell].top;
	float s2b[3], s2t[3];
	int d;
	for(d = 0; d < 3; d++) {
		s2b[d] = bot[d] - s[d];
		s2t[d] = top[d] - s[d];
	}
	return pluecker_(dir, &dist, s2b, s2t);
}
static void add(OctTree * tree, intptr_t ipar, intptr_t icell) {
	Cell * cell = &tree->pool[icell];
	if(cell->first_child != -1) {
		printf("never shall reach here, adding to a none leaf");
		return ;
	}
	tree->next[ipar] = cell->head_par;
	cell->head_par = ipar;
	cell->npar++;
}
static int full(OctTree * tree, intptr_t icell) {
	return tree->pool[icell].npar >= 16;
}
static intptr_t parent(OctTree * tree, intptr_t icell) {
	return tree->pool[icell].parent;
}
static intptr_t find(OctTree * tree, float pos[3], intptr_t icell) {
	Cell * cell = &tree->pool[icell];
	if(cell->first_child == -1) return icell;
	int i;
	for(i = 0; i < 8; i++) {
		if(inside(tree, pos, cell->first_child+i)) {
			return find(tree, pos, cell->first_child + i);
		}
	}
	printf("never reach here makesure inside() is ensured before find()");
	return -1;
}
static int split(OctTree * tree, intptr_t icell) {
	static int bitmask[] = {0x1, 0x2, 0x4};
	Cell * cell = &tree->pool[icell];
	if(tree->pool_length +8 >=tree->pool_size) {
		return 0;
	}
	cell->first_child = tree->pool_length;
	tree->pool_length += 8;
	Cell * fc = &tree->pool[cell->first_child];
	float center[3];
	int d;
	int i;
	for(d = 0; d < 3; d++) {
		center[d]  = 0.5 * (cell->top[d] + cell->bot[d]);
	}
	for(i = 0; i < 8; i++) {
		fc[i].parent = icell;
		for(d = 0; d < 3; d++) {
			fc[i].bot[d] = (bitmask[d] & i)?center[d]:cell->bot[d];
			fc[i].top[d] = (bitmask[d] & i)?cell->top[d]:center[d];
		}
		fc[i].head_par = -1;
		fc[i].first_child = -1;
		fc[i].npar = 0;
	}
	cell->npar = 0;
	intptr_t ipar;
	intptr_t nextpar;
	for(ipar = cell->head_par; ipar != -1; ipar = nextpar) {
		nextpar = tree->next[ipar];
		float pos[3];
		getpos(tree, pos, ipar);
		for(i = 0; i < 8; i++) {
			if(inside(tree, pos, i + cell->first_child)) {
				add(tree, ipar, i + cell->first_child);
			}
		}
	}
	cell->head_par = -1;
	return 1;
}
static int inside(OctTree * tree, float pos[3], intptr_t icell) {
	int d;
	Cell * cell = &tree->pool[icell];
	for(d = 0; d < 3; d++) {
		if(pos[d] >= cell->top[d] ||
			pos[d] < cell->bot[d]) return 0;
	}
	return 1;
}

static intptr_t sibling(OctTree * tree, intptr_t icell) {
	intptr_t parent = tree->pool[icell].parent;
	if(parent == -1) return -1;
	int ichild = icell - tree->pool[parent].first_child;
	if(ichild == 7) return -1;
	else return icell + 1;
}
static void getpos(OctTree * tree, float pos[3], intptr_t ipar) {
	int d;
	for(d = 0; d < 3; d++) {
		pos[d] = * ((float*) PyArray_GETPTR2(tree->locations, ipar, d));
	}
}
static float getsml(OctTree * tree, intptr_t ipar) {
	return * ((float*) PyArray_GETPTR1(tree->sml, ipar));
}
