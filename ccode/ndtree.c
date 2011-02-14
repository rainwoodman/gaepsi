#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>

#define HIDDEN __attribute__ ((visibility ("hidden")))  

#define INDEX_T int
#define DEFAULT_THRESHOLD 32
#define POOL_SIZE (1024*1024)
#define MAX_NODES (32*1024*1024)
#define MAX_DEPTH 32
#define BAR fprintf(stderr, "hit bar %s:%d\n", __FILE__, __LINE__);

#define DMAX 3

typedef struct _NDTree NDTree;
typedef struct _TreeNode TreeNode;

struct _TreeNode {
	float topleft[DMAX];
	float bottomright[DMAX];
	TreeNode * children;
	TreeNode * parent;
	INDEX_T * indices;
	int indices_size;
	int indices_length;
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
	int depth;
	int periodical;
	int dim;
	int node_count;
	int max_indices_length;
	int indices_threshold;
	int nodes_threshold;
	
};
static TreeNode * TreeNode_get_children(TreeNode * node) {
	return node->children;
}
static void TreeNode_init(NDTree * tree, TreeNode * node, 
	float topleft[], float bottomright[]) {
	if(topleft != NULL && bottomright != NULL) {
		memcpy(node->topleft, topleft, sizeof(float) * tree->dim);
		memcpy(node->bottomright, bottomright, sizeof(float) * tree->dim);
	}
	node->children = NULL;
	node->indices = NULL;
	node->indices_length = 0;
	node->indices_size = 0;
	node->depth = 0;
	node->sml_max = 0.0;
	tree->node_count++;
}
static int TreeNode_isleaf(TreeNode * node) {
	return node->children == NULL;
}
static void TreeNode_clear(NDTree * tree, TreeNode * node) {
	int i;
	if(!TreeNode_isleaf(node)) {
		for(i = 0; i < (1 << tree->dim); i++) {
			TreeNode_clear(tree, &(node->children[i]));
		}
		PyMem_Del(node->children);
	} else {
		PyMem_Del(node->indices);
		node->indices_length = 0;
		node->indices_size = 0;
		node->indices = NULL;
	}
	tree->node_count--;
}
static void TreeNode_append(TreeNode * node, INDEX_T index) {
	if(node->indices_size == 0) {
		node->indices_size = 8;
		node->indices = PyMem_New(INDEX_T, node->indices_size);
	}
	if(node->indices_size == node->indices_length) {
		node->indices_size *= 2;
		node->indices = PyMem_Resize(node->indices, INDEX_T, node->indices_size);
	}
	node->indices[node->indices_length] = index;
	node->indices_length ++;
}

static int TreeNode_is_inside(NDTree * tree, TreeNode * node, float pos[]) {
	/* see if a pos is inside of a node */
	float * boxsize = tree->boxsize;
	int d;
	for(d = 0; d < tree->dim; d++) {
		if(pos[d] < node->topleft[d]) return 0;
		if(pos[d] >= node->bottomright[d]) return 0;
	}
	return 1;
}

static int TreeNode_is_inside_sml(NDTree * tree, TreeNode * node, float pos[]) {
	/* see if a pos is inside of a node, sml into account */
	float * boxsize = tree->boxsize;
	int d;
	float sml = node->sml_max;
	for(d = 0; d < tree->dim; d++) {
	/*FIXME: periodical */
		if(pos[d] < node->topleft[d] - sml) return 0;
		if(pos[d] > node->bottomright[d] + sml) return 0;
	}
	return 1;
}


/* returns the min child node length */
static void TreeNode_split(NDTree * tree, TreeNode * node) {
	static int bitmask[DMAX] = { 0x1, 0x2, 0x4};
	int i, d;
	if(!TreeNode_isleaf(node)) {
		fprintf(stderr, "%p not a leaf\n", node);
		return;
	}
	float w2[DMAX];
	for(d = 0; d < tree->dim; d++) {
		w2[d] = (node->bottomright[d] - node->topleft[d])* 0.5;
		float t = tree->boxsize[d] / (1 << node->depth) * 0.5;
		if(fabs(t - w2[d]) > 1e-3) {
			printf("t %g!= w2%g\n", t, w2[d]);
		}
	}

	node->children = PyMem_New(TreeNode, (1<<tree->dim));

	for(i = 0; i < (1<<tree->dim); i++) {
		TreeNode * child = &(node->children[i]);
		float topleft[DMAX];
		float bottomright[DMAX];
		int p;
		for(d = 0; d < tree->dim; d++) {
			int pos = (i & bitmask[d]);
			if(pos == 0) {
				topleft[d] = node->topleft[d];
				bottomright[d] = node->topleft[d] + w2[d];
			} else {
				topleft[d] = node->topleft[d] + w2[d];
				bottomright[d] = node->bottomright[d];
			}
		}
		TreeNode_init(tree, child, topleft, bottomright);

		child->parent = node;
		child->depth = node->depth + 1;
		if(child->depth > tree->depth) {
			tree->depth = child->depth;
		}
		int count = 0;
		for(p = 0; p < node->indices_length; p++) {
			int d;
			float pos[DMAX];
			INDEX_T index = node->indices[p];
			for(d = 0; d < tree->dim; d++) {
				pos[d] = *((float*)PyArray_GETPTR2(tree->POS, index, d));
			}
			if(TreeNode_is_inside(tree, child, pos)) {
				TreeNode_append(child, index);
			}
		}
	}
	PyMem_Del(node->indices);
	node->indices_length = 0;
	node->indices_size = 0;
}

typedef struct __TreeNodeList {
	TreeNode * node;
	struct __TreeNodeList * next;
} TreeNodeList;

static TreeNodeList * TreeNode_find0(NDTree * tree, TreeNode * node, float pos[], TreeNodeList * tail) {
	if(!TreeNode_is_inside_sml(tree, node, pos)) return tail;
	if(!TreeNode_isleaf(node)) {
		int i;
		for(i = 0; i < (1 << tree->dim); i++) {
			tail = TreeNode_find0(tree, &(node->children[i]), pos, tail);
		}
	} else {
		tail-> node = node;
		tail->next = PyMem_New(TreeNodeList,1);
		tail->next->node = NULL;
		tail = tail->next;
		tail->next = NULL;
	}
	return tail;
}
static TreeNodeList * TreeNode_find(NDTree * tree, TreeNode * node, float pos[]) {
	TreeNodeList * list = PyMem_New(TreeNodeList, 1);
	list->next = NULL;
	list->node = NULL;
	TreeNode_find0(tree, node, pos, list);
	return list;
}
static void TreeNodeList_free(TreeNodeList * list) {
	TreeNodeList * p=list->next; 
	TreeNodeList * q; 
	while(p != NULL) {
		q = p->next;
		PyMem_Del(p);
		p = q;
	}
}
static TreeNode * TreeNode_locate(NDTree * tree, TreeNode * node, float pos[]) {
/* find a tree leaf that contains point pos*/
	if(!TreeNode_is_inside(tree, node, pos)) return NULL;
	if(!TreeNode_isleaf(node)) {
		int i;
		for(i = 0; i < (1 << tree->dim); i++) {
			TreeNode * rt = TreeNode_locate(tree, &(node->children[i]), pos);
			if(rt) return rt;
		}
		return NULL;
	} else {
		return node;
	}
}
static TreeNode * TreeNode_locate_debug(NDTree * tree, TreeNode * node, float pos[]) {
/* find a tree leaf that contains point pos*/
	fprintf(stderr, "topleft %f %f %f, bottomright %f %f %f\n",
		node->topleft[0],
		node->topleft[1],
		node->topleft[2],
		node->bottomright[0],
		node->bottomright[1],
		node->bottomright[2]);
	if(!TreeNode_is_inside(tree, node, pos)) return NULL;
	if(!TreeNode_isleaf(node)) {
		int i;
		for(i = 0; i < (1 << tree->dim); i++) {
			TreeNode * rt = TreeNode_locate_debug(tree, &(node->children[i]), pos);
			if(rt) return rt;
		}
		return NULL;
	} else {
		return node;
	}
}

static float TreeNode_calc_sml(NDTree * tree, TreeNode * node) {
	int i;
	if(!TreeNode_isleaf(node)) {
		for(i = 0; i < (1 << tree->dim); i++) {
			float t = TreeNode_calc_sml(tree, &(node->children[i]));
			if(t > node->sml_max) node->sml_max = t;
		}
	} else {
		for(i = 0; i < node->indices_length; i++) {
			INDEX_T index = node->indices[i];
			float t = *((float*)PyArray_GETPTR1(tree->S,index));
			if(t > node->sml_max) node->sml_max = t;
		}
		if(node->indices_length > tree->max_indices_length) tree->max_indices_length = node->indices_length;
	}
	return node->sml_max;
}
static void NDTree_dealloc(NDTree * self) {
	fprintf(stderr, "NDTree dispose %p refcount = %u\n", self, (unsigned int)self->ob_refcnt);
	TreeNode_clear(self, &(self->root));
	self->ob_type->tp_free((PyObject*) self);
}

static PyObject * NDTree_new(PyTypeObject * type, 
	PyObject * args, PyObject * kwds) {
	NDTree * self;
	self = (NDTree *)type->tp_alloc(type, 0);
	fprintf(stderr, "allocated a NDTree at %p\n", self);
	TreeNode_init(self, &(self->root), NULL, NULL);
	return (PyObject *) self;
}

static int NDTree_init(NDTree * self, 
	PyObject * args, PyObject * kwds) {
	PyArrayObject * POS;
	PyArrayObject * S;
	PyArrayObject * boxsize;
	PyArrayObject * origin;
	INDEX_T length = 0;
	int periodical = 1;
	INDEX_T index;
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
	self->indices_threshold = DEFAULT_THRESHOLD;
	self->depth = 0;
	self->nodes_threshold = MAX_NODES;
	self->periodical = periodical;
	float topleft[DMAX];
	float w[DMAX];
	for(d = 0; d < dim ; d++) {
		topleft[d] = self->origin[d];
		w[d] = self->boxsize[d];
	}
	TreeNode_init(self, &(self->root), topleft, w);

	fprintf(stderr, "handling %d particles\n", length);

	TreeNode * last_node = &(self->root);

	for(index = 0; index < length; index++) {
		int d;
		float pos[DMAX];
		for(d = 0; d < self->dim; d++) {
			pos[d] = *((float*)PyArray_GETPTR2(POS, index, d));
		}
		/*locality saves us some lookups*/
		TreeNode * node = TreeNode_locate(self, last_node, pos);
		while(node == NULL && last_node->parent) {
			last_node = last_node->parent;
			node = TreeNode_locate(self, last_node, pos);
		}
		while(node != NULL && self->node_count < self->nodes_threshold && self->depth < MAX_DEPTH && node->indices_length >= self->indices_threshold) {
			TreeNode_split(self, node);
			node = TreeNode_locate(self, node, pos);
		}
		if(node == NULL) {
			fprintf(stderr, "Warning: pos (%f %f %f) not inserted\n", pos[0], pos[1], pos[2]);
			TreeNode_locate_debug(self, &(self->root), pos);
			continue;
		}
		TreeNode_append(node, index);
		last_node = node;
	}
	TreeNode_calc_sml(self, &(self->root));
	Py_DECREF(boxsize);
	Py_DECREF(origin);
	Py_DECREF(self->POS);
	Py_DECREF(self->S);
	return 0;
}

static PyObject * NDTree_str(NDTree * self) {
	return PyString_FromFormat(
	"D=%d %p(%d bytes), nodes=%d(%d), depth=%d(%d), max_indices_length=%d(%d), periodical=%d, 1000*sml=%d",
	self->dim, self, self->node_count * sizeof(TreeNode),
	self->node_count, self->nodes_threshold, self->depth, MAX_DEPTH,
	self->max_indices_length, self->indices_threshold, self->periodical, (int)(self->root.sml_max * 1000));
}

static PyObject * NDTree_list(NDTree * self, 
	PyObject * args, PyObject * kwds) {
	float pos[3] = {0};
	static char * kwlist[] = {"x", "y", "z", NULL};
	if(! PyArg_ParseTupleAndKeywords(args, kwds, "ff|f", kwlist,
		&pos[0], &pos[1], &pos[2])) {
		return NULL;
	}
	TreeNodeList * nodes = TreeNode_find(self, &(self->root), pos);
	npy_intp dims[] = {0};

	INDEX_T size = 0;
	TreeNodeList * p;
	for(p = nodes; p!=NULL; p = p->next) {
		if(p->node == NULL) continue;
		size += p->node->indices_length;
	}
	
	dims[0] = size;
    PyObject * list = NULL;
	if(sizeof(INDEX_T) == 4) {
		list = PyArray_EMPTY(1, dims, NPY_INT32, 0);
	} else {
		list = PyArray_EMPTY(1, dims, NPY_INT64, 0);
	}
	INDEX_T * pointer = PyArray_DATA(list);
	for(p = nodes; p!=NULL; p = p->next) {
		if(p->node == NULL) continue;
		memcpy(pointer, p->node->indices, sizeof(INDEX_T) * p->node->indices_length);
		pointer += p->node->indices_length;
	}
	TreeNodeList_free(nodes);
	return list;
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
		"keywords: x, y, z. returns a list\n"
		"plist is a list of particle indices that may contribute to the give position\n"
		},
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
