import numpy as np

def gen_reader(reader):
  return "\n\n".join(
    [ gen_header_struct(reader),
      gen_blockid_func(reader),
      gen_blockname_func(reader),
      gen_itemsize_func(reader),
      gen_pstart_func(reader),
      gen_length_func(reader),
      gen_offset_func(reader),
      gen_def_header_func(reader),
      gen_get_constant_funcs(reader),
      gen_set_constant_funcs(reader),
      gen_read_func(reader),
      gen_write_func(reader), "\n"]);

def gen_itemsize_func(reader):
  return "\n".join([
    "static size_t _itemsize(const char * blk, char ** error) {",
    "\n".join([
       " ".join([
       "\tif(!strcmp(blk, \"%s\"))" % schema['name'],
       "return",
       "%d" % schema['dtype'].itemsize,
       ";"
       ]) for schema in reader.schemas]
     ),
     "\tif(!strcmp(blk, \"header\")) return %d;" % reader.header_dtype.itemsize,
     "\tasprintf(error, \"block %s is unknown\", blk);",
     "\treturn (size_t)-1;"
    "}"])

def gen_length_func(reader):
  return "\n".join([
    """
static size_t _length(const struct header_t * h, const char * blk, char ** error) {
""",
    "\n".join([
     "\n".join([
"""
	if(!strcmp(blk, "%s")) {
""" % schema['name'],
"""
		if(%s) {
""" % "&&".join(["1"] + ["h->%s" % field for field in schema['conditions']]),
"""
			return %s;
""" % "+".join(["(size_t)(h->N[%d])" % ptype for ptype in schema['ptypes']]),
"""
		} else {
			return 0;
		}
	}
"""]) for schema in reader.schemas]),
"""
	if(!strcmp(blk, "header")) return 1;
	asprintf(error, "block %s is unknown", blk);
	return (size_t)-1;
}
"""])

def gen_pstart_func(reader):
  return "\n".join([
    "static size_t _pstart(const struct header_t * h, const char * blk, int ptype, char ** error) {",
    "\n".join([
       " ".join([
       "\tif(!strcmp(blk, \"%s\"))" % schema['name'],
       "return",
       "+".join(["(%d < ptype)?(size_t)(h->N[%d]): 0" % (ptype, ptype) for ptype in schema['ptypes']]),
       ";"
       ]) for schema in reader.schemas]
     ),
     "\tasprintf(error, \"block %s is unknown/unstable for _pstart\", blk);",
     "\treturn (size_t)-1;"
    "}"])

def gen_blockid_func(reader):
  return "\n".join([
    "static int _blockid(const char * blk, char ** error) {",
    "\n".join([
       " ".join([
       "\tif(!strcmp(blk, \"%s\"))" % schema['name'],
       "return",
       "%d" % (id + 1),
       ";"
       ]) for (id, schema) in zip(range(len(reader.schemas)), reader.schemas)],
     ),
     "\tif(!strcmp(blk, \"header\")) return 0;",
     "\tasprintf(error, \"block %s is unknown\", blk);",
     "\treturn -1;"
    "}"])

def gen_blockname_func(reader):
  return "\n".join([
    "static const char * _blockname(const int blockid, char ** error) {",
       "\tstatic char * names[] = {\"header\", ",
       ",".join(["".join(["\"", schema['name'], "\""])
              for schema in reader.schemas]),
       "};",
     """
	if(blockid < 0 || blockid >= sizeof(names) / sizeof(char*)) {
		asprintf(error, \"blockid %d is unknown\", blockid);
		return NULL;
	}
	return names[blockid];
     """,
    "}"])

#! FIXME: shall use reader.constants dictionary.
def gen_get_constant_funcs(reader):
  def line(f):
    if f == "N" or f == "Ntot" or f == "mass": return ""
    if hasattr(reader.constants[f], "isalnum"):
      return "c->%s = h->%s;" %(f, reader.constants[f])
    else :
      return "c->%s = %s;" %(f, str(reader.constants[f]))

  return "\n".join(["""
static void _get_constants(const struct header_t * h, struct constants_t * c) {
	int i;
	for(i = 0; i< 6; i++) {
		c->N[i] = h->N[i];
		c->Ntot[i] = h->Nparticle_total_low[i]
			+ (((size_t) h->Nparticle_total_high[i]) << 32);
		c->mass[i] = h->mass[i];
	}
""",
    "\n".join([line(f) for f in reader.constants]),
"""
}
"""])

def gen_set_constant_funcs(reader):
  def line(f):
    if f == "N" or f == "Ntot" or f == "mass": return ""
    if hasattr(reader.constants[f], "isalnum"):
      return "h->%s = c->%s;" %(reader.constants[f], f)
    else :
      return ""
  return "\n".join(["""
static void _set_constants(struct header_t * h, const struct constants_t * c) {
	int i;
	for(i = 0; i< 6; i++) {
		h->N[i] = c->N[i];
		h->Nparticle_total_low[i] = c->Ntot[i];
		h->Nparticle_total_high[i] = c->Ntot[i] >> 32;
		h->mass[i] = c->mass[i];
	}
""",
    "\n".join([line(f) for f in reader.constants]),
"""
}
"""])
   

def gen_def_header_func(reader):
  return "\n".join([
    "static void _def_header(struct header_t * h) {",
    "\n".join([
      "h->%s = %s;" %( d, str(reader.defaults[d])) for
      d in reader.defaults
    ]),
    "}"])


def gen_offset_func(reader):
  return """
static size_t _offset(const struct header_t * h, const char * blk, char ** error) {
	int id = _blockid(blk, error);
	int i;
	if(*error) return (size_t) -1;
	size_t offset = 0;
	for(i = 0; i < id; i++) {
/* we are sure there won't be errors here*/
		const char * blknm = _blockname(i, error);
		size_t bsize = _length(h, blknm, error) * _itemsize(blknm, error);
		if(bsize > 0) offset += (bsize + 8);
	}
	return offset;
}
  """

def gen_read_func(reader):
  return """
static void _read(struct header_t * h, const char * blk, void * buffer, int start, int length, FILE * fp, char ** error) {
	size_t off = _offset(h, blk, error);
	if(*error) return;
	size_t l = _length(h, blk, error);
	size_t b = _itemsize(blk, error);
	if(l == 0) return;
	size_t bsize = l * b;
	fseek(fp, off, SEEK_SET);
	int blksize = 0;
	fread(&blksize, sizeof(int), 1, fp);
	if(blksize != (int)bsize) {
		asprintf(error, "block start size of %s mismatch; file says %u, reader says %lu\\n",
				blk, blksize, bsize);
		return;
	}
	fseek(fp, start * b, SEEK_CUR);
	if(!strcmp(blk, "header")) buffer = h;
	fread(buffer, b, length, fp);
	fseek(fp, (l - length - start) * b, SEEK_CUR);
	blksize = 0;
	fread(&blksize, sizeof(int), 1, fp);
	if(blksize != (int)bsize) {
		asprintf(error, "block end size of %s mismatch; file says %u, reader says %lu\\n",
				blk, blksize, bsize);
		return;
	}
}
  """

def gen_write_func(reader):
  return """
static void _write(const struct header_t * h, const char * blk, const void * buffer, int start, int length, FILE * fp, char ** error) {
	size_t off = _offset(h, blk, error);
	if(*error) return;
	size_t l = _length(h, blk, error);
	if(l == 0) return;
	size_t b = _itemsize(blk, error);
    size_t bsize = l * b;
	fseek(fp, off, SEEK_SET);
	int blksize = bsize;
	if(!strcmp(blk, "header")) buffer = h;
	fwrite(&blksize, sizeof(int), 1, fp);
	fseek(fp, start * b, SEEK_CUR);
	fwrite(buffer, b, length, fp);
	fseek(fp, (l - length - start) * b, SEEK_CUR);
	blksize = bsize;
	fwrite(&blksize, sizeof(int), 1, fp);
}
  """

def gen_header_struct(reader):
  return cstruct(reader.header_dtype, "header_t")

def cdecl(dt, varname):
  base = dt.base
  if base == np.int64:
    ctype = "long long"
  elif base == np.uint64:
    ctype = "unsigned long long"
  elif base == np.int32:
    ctype = "int"
  elif base == np.uint32:
    ctype = "unsigned int"
  elif base == np.int16:
    ctype = "short int"
  elif base == np.uint16:
    ctype = "unsigned short int"
  elif base == np.int8:
    ctype = "char"
  elif base == np.uint8:
    ctype = "unsigned char"
  elif base == np.float32:
    ctype = "float"
  elif base == np.float64:
    ctype = "double"
  else:
    raise ValueError("type unknown", base, np.uint32)

  shape = dt.shape
  if not shape: 
    return " ".join([ctype, varname])
  else:
    inds = "".join(["[%d]" % d for d in shape])
    return "".join([ctype, " ", varname, inds])
def cstruct(dt, struname):
  fields = [cdecl(dt.fields[name][0], name) for name in dt.names]
  inner = ";\n".join(fields)
  return "".join(["struct ", struname, "{\n", inner, ";\n};"])

