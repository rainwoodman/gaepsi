#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>

#define HIDDEN __attribute__ ((visibility ("hidden")))  

#define INDEX_T int
#define DEFAULT_THRESHOLD 32
#define MAX_NODES (32*1024*1024)
#define MAX_DEPTH 32
#define BAR fprintf(stderr, "hit bar %s:%d\n", __FILE__, __LINE__);
#define INITIAL_POOL_SIZE (1024 * 1024)

#define DMAX 3
#define NODEINDEX_T int

#define FOR_CHILDREN(tree, node, i, child)  int (i); TreeNode* (child); \
		for((i) = 0, (child)=GET_NODE((tree), (node)->children + (i)); \
			(i) < (1 << (tree)->dim); \
			(i)++, (child)=GET_NODE((tree), (node)->children + (i)))

typedef struct _NDTree NDTree;
typedef struct _TreeNode TreeNode;

struct _TreeNode {
	float topleft[DMAX];
	float bottomright[DMAX];
	NODEINDEX_T children;
	NODEINDEX_T parent;
	INDEX_T * indices;
	int indices_size;
	int indices_length;
	short int depth;
	short int leaf;
	float sml_max;
#define IS_LEAF(node) ((node)->leaf)
#define IS_ROOT(tree, node) ((node) == (tree)->pool)
};

struct _NDTree {
	PyObject_HEAD
	PyArrayObject * POS;
	PyArrayObject * SML;
	float boxsize[DMAX];
	float origin[DMAX];
	int depth;
	int periodical;
	int dim;
	int max_indices_length;
	int indices_threshold;
	int nodes_threshold;
	TreeNode * pool;
	size_t pool_size;
	size_t pool_length;
#define GET_NODE(tree, index) (&((tree)->pool[(index)]))
};

static void TreeNode_init(NDTree * tree, TreeNode * node, 
	float topleft[], float bottomright[]) {
	memcpy(node->topleft, topleft, sizeof(float) * tree->dim);
	memcpy(node->bottomright, bottomright, sizeof(float) * tree->dim);
	node->indices = NULL;
	node->indices_length = 0;
	node->indices_size = 0;
	node->depth = 0;
	node->sml_max = 0.0;
	node->leaf = 1;
	node->children = -1;
	node->parent = -1;
}
static void TreeNode_clear(NDTree * tree, NODEINDEX_T nodeindex) {
	TreeNode * node = &(tree->pool[nodeindex]);
	if(!IS_LEAF(node)) {
		FOR_CHILDREN(tree, node, i, child) {
			TreeNode_clear(tree, node->children + i);
		}
	} else {
		PyMem_Del(node->indices);
		node->indices_length = 0;
		node->indices_size = 0;
		node->indices = NULL;
	}
}
static void TreeNode_append(TreeNode * node, INDEX_T index) {
	if(node->indices_size == 0) {
		node->indices_size = 8;
		node->indices = PyMem_New(INDEX_T, node->indices_size);
	}
	if(node->indices_size == node->indices_length) {
		node->indices_size += 8;
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


/* returns a pointer to the splited node, which may be at a different memory location */ 
static TreeNode * TreeNode_split(NDTree * tree, TreeNode * node) {
	static int bitmask[DMAX] = { 0x1, 0x2, 0x4};
	int d;
	if(!IS_LEAF(node)) {
		fprintf(stderr, "%p not a leaf\n", node);
		return;
	}
	float w2[DMAX];
	for(d = 0; d < tree->dim; d++) {
		w2[d] = (node->bottomright[d] - node->topleft[d])* 0.5;
	}

	NODEINDEX_T nodeindex = node - tree->pool;
	node = GET_NODE(tree, nodeindex);
	node->children = tree->pool_length;
	node->leaf = 0;
	tree->pool_length += 1 << tree->dim;
	if(tree->pool_length > tree->pool_size) {
		tree->pool_size += INITIAL_POOL_SIZE;
		tree->pool = PyMem_Resize(tree->pool, TreeNode, tree->pool_size);
		node = GET_NODE(tree, nodeindex);
	}
	FOR_CHILDREN(tree, node, i, child) {
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
		child->parent = node - tree->pool; /* get the offset */
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
	return node;
}

typedef struct __TreeNodeList {
	TreeNode * node;
	struct __TreeNodeList * next;
} TreeNodeList;

static TreeNodeList * TreeNode_find0(NDTree * tree, TreeNode * node, float pos[], TreeNodeList * tail) {
	if(!TreeNode_is_inside_sml(tree, node, pos)) return tail;
	if(!IS_LEAF(node)) {
		FOR_CHILDREN(tree, node, i, child) {
			tail = TreeNode_find0(tree, child, pos, tail);
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
	if(!IS_LEAF(node)) {
		FOR_CHILDREN(tree, node, i, child) {
			TreeNode * rt = TreeNode_locate(tree, child, pos);
			if(rt) return rt;
		}
		return NULL;
	} else {
		return node;
	}
}
TreeNode * TreeNode_locate_debug(NDTree * tree, TreeNode * node, float pos[]) {
/* find a tree leaf that contains point pos*/
	fprintf(stderr, "topleft %f %f %f, bottomright %f %f %f\n",
		node->topleft[0],
		node->topleft[1],
		node->topleft[2],
		node->bottomright[0],
		node->bottomright[1],
		node->bottomright[2]);
	if(!TreeNode_is_inside(tree, node, pos)) return NULL;
	if(!IS_LEAF(node)) {
		FOR_CHILDREN(tree, node, i, child) {
			TreeNode * rt = TreeNode_locate_debug(tree, child, pos);
			if(rt) return rt;
		}
		return NULL;
	} else {
		return node;
	}
}

static float TreeNode_calc_sml(NDTree * tree, TreeNode * node) {
	int i;
	if(!IS_LEAF(node)) {
		FOR_CHILDREN(tree, node, i, child) {
			float t = TreeNode_calc_sml(tree, child);
			if(t > node->sml_max) node->sml_max = t;
		}
	} else {
		for(i = 0; i < node->indices_length; i++) {
			INDEX_T index = node->indices[i];
			float t = *((float*)PyArray_GETPTR1(tree->SML,index));
			if(t > node->sml_max) node->sml_max = t;
		}
		if(node->indices_length > tree->max_indices_length) tree->max_indices_length = node->indices_length;
	}
	return node->sml_max;
}
static void NDTree_dealloc(NDTree * self) {
	fprintf(stderr, "NDTree dispose %p refcount = %u %p\n", self, (unsigned int)self->ob_refcnt, self->pool);
	if(self->pool != NULL) {
		TreeNode_clear(self, 0);
		PyMem_Del(self->pool);
	}
	self->ob_type->tp_free((PyObject*) self);
}

static PyObject * NDTree_new(PyTypeObject * type, 
	PyObject * args, PyObject * kwds) {
	NDTree * self;
	self = (NDTree *)type->tp_alloc(type, 0);
	fprintf(stderr, "allocated a NDTree at %p\n", self);
	return (PyObject *) self;
}

static int NDTree_init(NDTree * self, 
	PyObject * args, PyObject * kwds) {
	PyArrayObject * POS = NULL;
	PyArrayObject * SML = NULL;
	PyArrayObject * boxsize;
	PyArrayObject * origin;
	INDEX_T length = 0;
	int periodical = 1;
	INDEX_T index;
	int d;
	int dim;
	static char * kwlist[] = {"D", "POS", "origin", "boxsize", "SML", NULL};
    fprintf(stderr, "NDTree_init on %p\n", self);
	if(! PyArg_ParseTupleAndKeywords(args, kwds, "iO!O!O!|O!", kwlist,
		&dim,
		&PyArray_Type, &POS,
		&PyArray_Type, &origin, 
		&PyArray_Type, &boxsize, 
		&PyArray_Type, &SML
	)) return -1;
	
	self->dim = dim;
	self->POS = (PyArrayObject*) PyArray_Cast(POS, NPY_FLOAT);

    boxsize = (PyArrayObject*) PyArray_Cast(boxsize, NPY_FLOAT);
    origin = (PyArrayObject*) PyArray_Cast(origin, NPY_FLOAT);

	length = PyArray_DIM((PyObject*)POS, 0);
	for(d = 0; d < dim; d++) {
		self->boxsize[d] = *((float*)PyArray_GETPTR1(boxsize, d));
		self->origin[d] = *((float*)PyArray_GETPTR1(origin, d));
	}
	self->pool_length = 0;
	self->indices_threshold = DEFAULT_THRESHOLD;
	self->depth = 0;
	self->nodes_threshold = MAX_NODES;
	self->periodical = periodical;
	self->pool = PyMem_New(TreeNode, INITIAL_POOL_SIZE);
	self->pool_size = INITIAL_POOL_SIZE;

	float topleft[DMAX];
	float w[DMAX];
	for(d = 0; d < dim ; d++) {
		topleft[d] = self->origin[d];
		w[d] = self->boxsize[d];
	}

	TreeNode_init(self, GET_NODE(self, 0), topleft, w);
	GET_NODE(self, 0)->leaf = 1;
	self->pool_length = 1;

	fprintf(stderr, "handling %d particles\n", length);

	TreeNode * last_node = GET_NODE(self, 0);

	for(index = 0; index < length; index++) {
		int d;
		float pos[DMAX];
		for(d = 0; d < self->dim; d++) {
			pos[d] = *((float*)PyArray_GETPTR2(POS, index, d));
		}
		/*locality saves us some lookups*/
		TreeNode * node = TreeNode_locate(self, last_node, pos);
		while(node == NULL && !IS_ROOT(self, last_node)) {
			last_node = GET_NODE(self, last_node->parent);
			node = TreeNode_locate(self, last_node, pos);
		}
		while(node != NULL && self->pool_length < self->nodes_threshold && self->depth < MAX_DEPTH && node->indices_length >= self->indices_threshold) {
		/* split may reallocate the nodes, thus invalidating last_node. luckily we no longer need it after this point.*/
			node = TreeNode_split(self, node);
			node = TreeNode_locate(self, node, pos);
		}
		if(node == NULL) {
			fprintf(stderr, "Warning: pos (%f %f %f) not inserted\n", pos[0], pos[1], pos[2]);
			TreeNode_locate_debug(self, GET_NODE(self, 0), pos);
			continue;
		}
		TreeNode_append(node, index);
		last_node = node;
	}

	if(SML != NULL) {
		self->SML = (PyArrayObject*) PyArray_Cast(SML, NPY_FLOAT);
		TreeNode_calc_sml(self, GET_NODE(self, 0));
		Py_DECREF(self->SML);
	}
	Py_DECREF(boxsize);
	Py_DECREF(origin);
	Py_DECREF(self->POS);
	return 0;
}

static PyObject * NDTree_str(NDTree * self) {
	return PyString_FromFormat(
	"D=%d %p(%d bytes), nodes=%d(<%d), depth=%d(%d), max_indices_length=%d(%d), periodical=%d, 1000*sml=%d",
	self->dim, self, self->pool_size * sizeof(TreeNode),
	self->pool_length, self->nodes_threshold, self->depth, MAX_DEPTH,
	self->max_indices_length, self->indices_threshold, self->periodical, (int)(GET_NODE(self, 0)->sml_max * 1000));
}

static PyObject * NDTree_list(NDTree * self, 
	PyObject * args, PyObject * kwds) {
	float pos[3] = {0};
	static char * kwlist[] = {"x", "y", "z", NULL};
	if(! PyArg_ParseTupleAndKeywords(args, kwds, "ff|f", kwlist,
		&pos[0], &pos[1], &pos[2])) {
		return NULL;
	}
	TreeNodeList * nodes = TreeNode_find(self, GET_NODE(self, 0), pos);
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
