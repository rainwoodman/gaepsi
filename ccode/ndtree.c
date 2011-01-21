#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>

#define HIDDEN __attribute__ ((visibility ("hidden")))  

#define INDEX_T int
#define DEFAULT_THRESHOLD 8192
#define MAX_DEPTH 8
#define BAR fprintf(stderr, "hit bar %s:%d\n", __FILE__, __LINE__);
#define STR0(x) #x
#define STR(x) STR0(x)

#define DMAX 3

typedef struct _NDTree NDTree;
typedef struct _TreeNode TreeNode;

struct _TreeNode {
	float topleft[DMAX];
	float width[DMAX];
	TreeNode * children[1 << DMAX];
	TreeNode * parent;
	INDEX_T * indices;
	int indices_size;
	int indices_length;
	NDTree * tree;
	int depth;
	float sml_max;
};

struct _NDTree {
	PyObject_HEAD
	PyArrayObject * POS;
	PyArrayObject * S;
	float boxsize[DMAX];
	float origin[DMAX];
	TreeNode root;
	int node_count;
	int depth;
	int periodical;
	int threshold;
	int max_depth;
	int dim;
};


static void TreeNode_init(TreeNode * node, NDTree * tree,
	float topleft[], float width[]) {
	if(topleft != NULL && width != NULL) {
		memcpy(node->topleft, topleft, sizeof(float) * tree->dim);
		memcpy(node->width, width, sizeof(float) * tree->dim);
	}
	node->children[0] = NULL;
	node->tree = tree;
	node->depth = 0;
	node->sml_max = 0.0;
	tree->node_count++;
}
static int TreeNode_isleaf(TreeNode * node) {
	return node->children[0] == NULL;
}
static void TreeNode_clear(TreeNode * node) {
	int i;
	if(!TreeNode_isleaf(node)) {
		for(i = 0; i < (1 << node->tree->dim); i++) {
			TreeNode_clear(node->children[i]);
			free(node->children[i]);
			node->children[i] = NULL;
		}
	} else {
		free(node->indices);
		node->indices_length = 0;
		node->indices_size = 0;
		node->indices = NULL;
	}
	node->tree->node_count--;
}
static void TreeNode_setchild(TreeNode * node, int chindex, TreeNode * child) {
	node->children[chindex] = child;
	child->parent = node;
    child->depth = node->depth + 1;
}
static void TreeNode_append(TreeNode * node, INDEX_T index) {
	if(node->indices_size == 0) {
		node->indices_size = 64;
		node->indices = calloc(sizeof(INDEX_T), node->indices_size);
	}
	if(node->indices_size == node->indices_length) {
		node->indices_size *= 2;
		node->indices = realloc(node->indices, sizeof(INDEX_T) * node->indices_size);
	}
	node->indices[node->indices_length] = index;
	node->indices_length ++;
	float s = *((float*)PyArray_GETPTR1(node->tree->S,index));
	if(s > node->sml_max) node->sml_max = s;
}

static int TreeNode_touch(TreeNode * node, float pos[]) {
	NDTree * tree = node->tree;
	int d;
	float * boxsize = tree->boxsize;
	float s = node->sml_max;
	
	for(d = 0; d < tree->dim; d++) {
		float w2 = node->width[d] * 0.5;
		float dist = fabs(pos[d] - node->topleft[d] - w2);
		if(tree->periodical && dist > 0.5 * boxsize[d]) dist = boxsize[d] - dist;
		if(dist > w2 + s) return 0;
	}
	return 1;
}
static int TreeNode_touch_i(TreeNode * node, INDEX_T index) {
	int d;
	NDTree * tree = node->tree;
	float pos[DMAX];
	for(d = 0; d < tree->dim; d++) {
		pos[d] = *((float*)PyArray_GETPTR2(tree->POS, index, d));
	}
//	float s = *((float*)PyArray_GETPTR1(tree->S,index));
	float * boxsize = tree->boxsize;
	
	for(d = 0; d < tree->dim; d++) {
		float w2 = node->width[d] * 0.5;
		float dist = fabs(pos[d] - node->topleft[d] - w2);
		if(tree->periodical && dist > 0.5 * boxsize[d]) dist = boxsize[d] - dist;
		if(dist > w2/* + s*/) return 0;
	}
	return 1;
}


/* returns the min child node length */
static int TreeNode_split(TreeNode * node) {
	static int bitmask[DMAX] = { 0x1, 0x2, 0x4};
	int i, d;
	NDTree * tree = node->tree;
	if(!TreeNode_isleaf(node)) {
		fprintf(stderr, "%p not a leaf\n", node);
		return 0;
	}
	float w2[DMAX];
	for(d = 0; d < tree->dim; d++) {
		w2[d] = node->width[d] * 0.5;
	}

	int max_child_length = 0;
	for(i = 0; i < (1<<tree->dim); i++) {
		TreeNode * child = calloc(sizeof(TreeNode), 1);
		float topleft[DMAX];
		int p;
		for(d = 0; d < tree->dim; d++) {
			topleft[d] = node->topleft[d] + ((i & bitmask[d]) >> d) * w2[d];
		}
		TreeNode_init(child, node->tree, topleft, w2);

		TreeNode_setchild(node, i, child);
		if(child->depth > node->tree->depth) {
			node->tree->depth = child->depth;
		}
		int count = 0;
		for(p = 0; p < node->indices_length; p++) {
			INDEX_T index = node->indices[p];
			if(TreeNode_touch_i(child, index)) {
				TreeNode_append(child, index);
				count ++;
			}
		}
		if(count > max_child_length) max_child_length = count;
	}
	free(node->indices);
	node->indices_length = 0;
	node->indices_size = 0;
	return max_child_length;
}

static void TreeNode_insert(TreeNode * node, INDEX_T index) {
	int i;
	if(!TreeNode_touch_i(node, index)) return;
	if(!TreeNode_isleaf(node)) {
		for(i = 0; i < (1 << node->tree->dim); i++) {
			TreeNode_insert(node->children[i], index);
		}
	} else {
		int threshold = node->tree->threshold;
		int max_depth = node->tree->max_depth;
		if(node->depth < max_depth && node->indices_length >= threshold) {
			TreeNode_split(node);
			for(i = 0; i < (1 << node->tree->dim); i++) {
				TreeNode_insert(node->children[i], index);
			}
		} else {
			TreeNode_append(node, index);
		}
	}
}

static TreeNode * TreeNode_find(TreeNode * node, float pos[]) {
	if(!TreeNode_touch(node, pos)) return NULL;
	if(!TreeNode_isleaf(node)) {
		int i;
		for(i = 0; i < (1 << node->tree->dim); i++) {
			TreeNode * rt = TreeNode_find(node->children[i], pos);
			if(rt) return rt;
		}
		return NULL;
	} else {
		return node;
	}
}
static void NDTree_dealloc(NDTree * self) {
	fprintf(stderr, "NDTree dispose %p refcount = %u\n", self, (unsigned int)self->ob_refcnt);
	TreeNode_clear(&(self->root));
	self->ob_type->tp_free((PyObject*) self);
}

static PyObject * NDTree_new(PyTypeObject * type, 
	PyObject * args, PyObject * kwds) {
	NDTree * self;
	self = (NDTree *)type->tp_alloc(type, 0);
	fprintf(stderr, "allocated a NDTree at %p\n", self);
	TreeNode_init(&(self->root), self, NULL, NULL);
	return (PyObject *) self;
}

static int NDTree_init(NDTree * self, 
	PyObject * args, PyObject * kwds) {
	PyArrayObject * POS;
	PyArrayObject * S;
	PyArrayObject * boxsize;
	PyArrayObject * origin;
	int length = 0;
	int periodical = 1;
	int i;
	int d;
	int dim;
	static char * kwlist[] = {"D", "POS", "SML", "origin", "boxsize", "periodical", NULL};
    fprintf(stderr, "NDTree_init on %p\n", self);
	if(! PyArg_ParseTupleAndKeywords(args, kwds, "iO!O!O!O!|i", kwlist,
		&dim,
		&PyArray_Type, &POS, 
		&PyArray_Type, &S,
		&PyArray_Type, &origin, 
		&PyArray_Type, &boxsize, 
        &periodical
	)) return -1;
	
	self->dim = dim;
	self->POS = (PyArrayObject*) PyArray_Cast(POS, NPY_FLOAT);
	self->S = (PyArrayObject*) PyArray_Cast(S, NPY_FLOAT);
    boxsize = (PyArrayObject*) PyArray_Cast(boxsize, NPY_FLOAT);
    origin = (PyArrayObject*) PyArray_Cast(origin, NPY_FLOAT);

	length = PyArray_Size((PyObject*)S);
	for(d = 0; d < dim; d++) {
		self->boxsize[d] = *((float*)PyArray_GETPTR1(boxsize, d));
		self->origin[d] = *((float*)PyArray_GETPTR1(origin, d));
	}
	self->node_count = 0;
	self->threshold = DEFAULT_THRESHOLD;
	self->depth = 0;
	self->max_depth = MAX_DEPTH;
	self->periodical = periodical;
	float topleft[DMAX];
	float w[DMAX];
	for(d = 0; d < dim ; d++) {
		topleft[d] = self->origin[d];
		w[d] = self->boxsize[d];
	}
	TreeNode_init(&(self->root), self, topleft, w);
	fprintf(stderr, "handling %d particles\n", length);
	for(i = 0; i < length; i++) {
		TreeNode_insert(&(self->root), i);
	}
	Py_DECREF(boxsize);
	Py_DECREF(origin);
	Py_DECREF(self->POS);
	Py_DECREF(self->S);
	return 0;
}

static PyObject * NDTree_str(NDTree * self) {
	return PyString_FromFormat(
	"D=%d %p, nodes=%d, depth=%d, threshold=%d, periodical=%d", 
	self->dim, self, 
	self->node_count, self->depth, 
	self->threshold, self->periodical);
}

static PyObject * NDTree_list(NDTree * self, 
	PyObject * args, PyObject * kwds) {
	float pos[3] = {0};
	static char * kwlist[] = {"x", "y", "z", NULL};
	if(! PyArg_ParseTupleAndKeywords(args, kwds, "ff|f", kwlist,
		&pos[0], &pos[1], &pos[2])) {
		return NULL;
	}
	TreeNode * node = TreeNode_find(&(self->root), pos);
	npy_intp dims[] = {0};
// FIXME: USE NPY_LONG if INDEX_T is long!
    PyObject * list = NULL;
	PyObject * rt = NULL;
	if(!node) {
		list = PyArray_EMPTY(1, dims, NPY_INT, 0);
	} else {
		dims[0] = node->indices_length;
		list = PyArray_SimpleNewFromData(1, dims, NPY_INT, node->indices);
	}
	rt = Py_BuildValue("Oi", list, node);
	return rt;
}

static PyTypeObject NDTreeType = {
	PyObject_HEAD_INIT(NULL)
	0, "ccode.NDTree",
	sizeof(NDTree)
};

static PyMemberDef NDTree_members[] = {
	{NULL}
};
static PyMethodDef NDTree_methods[] = {
	{"list", (PyCFunction) NDTree_list, 
		METH_KEYWORDS,
		"keywords: x, y, z. returns a (plist, key)\n"
		"plist is a list of particle indices that may contribute to the give position\n"
		"key is a unique hash key of the list\n"},
	{NULL}
};

void HIDDEN gadget_initNDTree (PyObject * m) {

	import_array();
	NDTreeType.tp_dealloc = (destructor) NDTree_dealloc;
	NDTreeType.tp_new = NDTree_new;
	NDTreeType.tp_init = (initproc) NDTree_init;
	NDTreeType.tp_str = (reprfunc) NDTree_str;
	NDTreeType.tp_members = NDTree_members;
	NDTreeType.tp_methods = NDTree_methods;
	NDTreeType.tp_doc = "NDTree(D, pos, sml, origin, boxsize, periodical=True)";
	NDTreeType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;

	if (PyType_Ready(&NDTreeType) < 0) return;

	Py_INCREF(&NDTreeType);
	PyModule_AddObject(m, "NDTree", (PyObject *) &NDTreeType);
}
