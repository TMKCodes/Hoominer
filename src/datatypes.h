#ifndef DATATYPES_H
#define DATATYPES_H
#include <stdio.h>

#define MAX 1000

typedef struct
{
  int data[MAX];
  int head;
  int tail;
  int count;
} IntFifo;

void init_int_fifo(IntFifo *q);
int enqueue_int_fifo(IntFifo *q, int value);
int dequeue_int_fifo(IntFifo *q, int *out);

#endif