#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>

#define INDEX_T int
#define D 2
#define DEFAULT_THRESHOLD 8192
#define BAR fprintf(stderr, "hit bar %s:%d\n", __FILE__, __LINE__);
typedef struct _QuadTree QuadTree;
typedef struct _TreeNode TreeNode;

struct _TreeNode {
	float topleft[D];
	float width[D];
	TreeNode * children[1 << D];
	TreeNode * parent;
	INDEX_T * indices;
	int indices_size;
	int indices_length;
	QuadTree * tree;
};

struct _QuadTree {
	PyObject_HEAD
	PyArrayObject * POS;
	PyArrayObject * S;
	float boxsize;
	TreeNode root;
	int node_count;
	int threshold;
};


void TreeNode_init(TreeNode * node, QuadTree * tree,
	float topleft[D], float width[D]) {
	memcpy(node->topleft, topleft, sizeof(float) * D);
	memcpy(node->width, width, sizeof(float) * D);
	node->children[0] = NULL;
	node->tree = tree;
	tree->node_count++;
}
int TreeNode_isleaf(TreeNode * node) {
	return node->children[0] == NULL;
}
void TreeNode_clear(TreeNode * node) {
	int i;
	if(!TreeNode_isleaf(node)) {
		for(i = 0; i < (1 << D); i++) {
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
void TreeNode_setchild(TreeNode * node, int chindex, TreeNode * child) {
	node->children[chindex] = child;
	child->parent = node;
}
void TreeNode_append(TreeNode * node, INDEX_T index) {
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
}

int TreeNode_touch(TreeNode * node, float pos[D]) {
	QuadTree * tree = node->tree;
	int d;
	float boxsize = tree->boxsize;
	float boxsize2 = 0.5 * tree->boxsize;
	
	for(d = 0; d < D; d++) {
		float w2 = node->width[d] * 0.5;
		float dist = fabs(pos[d] - node->topleft[d] - w2);
		if(dist > boxsize2) dist = boxsize - dist;
		if(dist > w2) return 0;
	}
	return 1;
}
int TreeNode_touch_i(TreeNode * node, INDEX_T index) {
	int d;
	QuadTree * tree = node->tree;
	float pos[D];
	for(d = 0; d < D; d++) {
		pos[d] = *((float*)PyArray_GETPTR2(tree->POS, index, d));
	}
	float s = *((float*)PyArray_GETPTR1(tree->S,index));
	float boxsize = tree->boxsize;
	float boxsize2 = 0.5 * tree->boxsize;
	
	for(d = 0; d < D; d++) {
		float w2 = node->width[d] * 0.5;
		float dist = fabs(pos[d] - node->topleft[d] - w2);
		if(dist > boxsize2) dist = boxsize - dist;
		if(dist > w2 + s) return 0;
	}
	return 1;
}


/* returns the max chld node length */
int TreeNode_split(TreeNode * node) {
	static int bitmask[5] = { 0x1, 0x2, 0x4, 0x8, 0x16};
	int i, d;

	if(!TreeNode_isleaf(node)) {
		fprintf(stderr, "%p not a leaf\n", node);
		return 0;
	}
	float w2[D];
	for(d = 0; d < D; d++) {
		w2[d] = node->width[d] * 0.5;
	}

	int max_child_length = 0;
	for(i = 0; i < (1<<D); i++) {
		TreeNode * child = calloc(sizeof(TreeNode), 1);
		float topleft[D];
		int p;
		for(d = 0; d < D; d++) {
			topleft[d] = node->topleft[d] + ((i & bitmask[d]) >> d) * w2[d];
		}
		TreeNode_init(child, node->tree, topleft, w2);

		TreeNode_setchild(node, i, child);
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

void TreeNode_insert(TreeNode * node, INDEX_T index) {
	if(!TreeNode_touch_i(node, index)) return;
	if(!TreeNode_isleaf(node)) {
		int i;
		for(i = 0; i < (1 << D); i++) {
			TreeNode_insert(node->children[i], index);
		}
	} else {
		int threshold = node->tree->threshold;
		if(node->indices_length >= threshold) {
			int length = TreeNode_split(node);
			if(length == threshold) {
				node->tree->threshold *= 2;
			}
			TreeNode_insert(node, index);
		} else {
			TreeNode_append(node, index);
		}
	}
}

TreeNode * TreeNode_find(TreeNode * node, float pos[D]) {
	if(!TreeNode_touch(node, pos)) return NULL;
	if(!TreeNode_isleaf(node)) {
		int i;
		for(i = 0; i < (1 << D); i++) {
			TreeNode * rt = TreeNode_find(node->children[i], pos);
			if(rt) return rt;
		}
		return NULL;
	} else {
		return node;
	}
}
static void QuadTree_dealloc(QuadTree * self) {
	TreeNode_clear(&(self->root));
	self->ob_type->tp_free((PyObject*) self);
}

static PyObject * QuadTree_new(PyTypeObject * type, 
	PyObject * args, PyObject * kwds) {
	QuadTree * self;
	self = (QuadTree *)type->tp_alloc(type, 0);
	return (PyObject *) self;
}

static int QuadTree_init(QuadTree * self, 
	PyObject * args, PyObject * kwds) {
	PyArrayObject * POS;
	PyArrayObject * S;
	float boxsize;
	int length = 0;
	int i;
	int d;
	static char * kwlist[] = {"POS", "S", "boxsize", NULL};
	if(! PyArg_ParseTupleAndKeywords(args, kwds, "O!O!f", kwlist,
		&PyArray_Type, &POS, 
		&PyArray_Type, &S,
		&boxsize
	)) return -1;
	
	self->POS = (PyArrayObject*) PyArray_Cast(POS, NPY_FLOAT);
	self->S = (PyArrayObject*) PyArray_Cast(S, NPY_FLOAT);
	length = PyArray_Size((PyObject*)S);
	self->boxsize = boxsize;
	self->node_count = 0;
	self->threshold = DEFAULT_THRESHOLD;
	float topleft[D];
	float w[D];
	for(d = 0; d < D ; d++) {
		topleft[d] = 0.0;
		w[d] = boxsize;
	}
	TreeNode_init(&(self->root), self, topleft, w);
	fprintf(stderr, "handling %d particles\n", length);
	for(i = 0; i < length; i++) {
		TreeNode_insert(&(self->root), i);
	}
	Py_DECREF(self->POS);
	Py_DECREF(self->S);
	return 0;
}

static PyObject * QuadTree_str(QuadTree * self) {
	return PyString_FromFormat("QuadTree %p, %d nodes, %d threshold", 
		self, self->node_count, self->threshold);
}

static PyObject * QuadTree_list(QuadTree * self, 
	PyObject * args, PyObject * kwds) {
	float pos[D + 10] = {0};
	static char * kwlist[] = {"x", "y", "z", NULL};
	if(! PyArg_ParseTupleAndKeywords(args, kwds, "ff|f", kwlist,
		&pos[0], &pos[1], &pos[2])) {
		return NULL;
	}
	TreeNode * node = TreeNode_find(&(self->root), pos);
	npy_intp dims[] = {0};
// FIXME: USE NPY_LONG if INDEX_T is long!
	if(!node) return PyArray_EMPTY(1, dims, NPY_INT, 0);
	dims[0] = node->indices_length;
	return PyArray_SimpleNewFromData(1, dims, NPY_INT, node->indices);
}

static PyTypeObject QuadTreeType = {
	PyObject_HEAD_INIT(NULL)
	0, "ccode.QuadTree",
	sizeof(QuadTree)
};

static PyMemberDef QuadTree_members[] = {
	{NULL}
};
static PyMethodDef QuadTree_methods[] = {
	{"list", (PyCFunction) QuadTree_list, 
		METH_KEYWORDS,
		"return a list of particle indices that may contribute to the give position"},
	{NULL}
};

static PyMethodDef module_methods[] = {
	{NULL}
};

void initquadtree(void) {
	PyObject * m;
	import_array();

	QuadTreeType.tp_dealloc = (destructor) QuadTree_dealloc;
	QuadTreeType.tp_new = QuadTree_new;
	QuadTreeType.tp_init = (initproc) QuadTree_init;
	QuadTreeType.tp_str = (reprfunc) QuadTree_str;
	QuadTreeType.tp_members = QuadTree_members;
	QuadTreeType.tp_methods = QuadTree_methods;

	QuadTreeType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;

	if (PyType_Ready(&QuadTreeType) < 0) return;

	m = Py_InitModule3("quadtree", module_methods, "QuadTree module");
	if (m == NULL) return;
	Py_INCREF(&QuadTreeType);
	PyModule_AddObject(m, "QuadTree", (PyObject *) &QuadTreeType);
}
