#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

typedef struct {
    int (*cmp_lt)(int i, int j, void* weight);
    void (*load)(int i, int j, void* data, void * weight);
    void * data;
    void * weight;
    size_t length;
} Heap;

void _siftdown(Heap * self, int startpos, int pos) {
    self->load(-1, pos, self->data, self->weight);
    while(pos > startpos) {
        int parent = (pos - 1) >> 1;
        if(self->cmp_lt(-1, parent, self->weight)) {
            self->load(pos, parent, self->data, self->weight);
            pos = parent;
            continue;
        }
        break;
    }
    self->load(pos, -1, self->data, self->weight);
}

void _siftup(Heap * self, int pos) {
    int startpos = pos;
    int childpos = 2 * pos + 1;
    self->load(-1, pos, self->data, self->weight);
    while(childpos < self->length) {
        int rightpos = childpos + 1;
        if(rightpos < self->length && 
                !(self->cmp_lt(childpos, rightpos, self->weight))) {
            childpos = rightpos;
        }
        self->load(pos, childpos, self->data, self->weight);
        pos = childpos;
        childpos = 2 * pos + 1;
    }
    self->load(pos, -1, self->data, self->weight);
    _siftdown(self, startpos, pos);
}



int cmp_lt(int i, int j, float * weight) {
    return (weight[i] < weight[j]);
}
void load(int i, int j, float * data, float * weight) {
    data[i] = data[j];
    weight[i] = weight[j];
}

void _heapify(Heap * self) {
    int i;
    for (i = self->length / 2 - 1; i >=0; i--) {
        _siftup(self, i);
    }
}

float data[9];
int main() {
    Heap wh;
    wh.cmp_lt = cmp_lt;
    wh.load = load;
    wh.data = data + 1;
    wh.weight = data + 1;
    wh.length = 8;
    int i;
    for (i = 0; i < wh.length; i++) {
      data[i+1] = wh.length - i;
    }
    _heapify(&wh);
    for (i = 0; i < wh.length; i++) {
      printf("%g\n", data[i+1]);
    }
    data[1] = 99;
    _siftup(&wh, 0);
    for (i = 0; i < wh.length; i++) {
      printf("%g\n", data[i+1]);
    }
    return 0;
}
