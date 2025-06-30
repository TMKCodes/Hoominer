#include "datatypes.h"

void init_int_fifo(IntFifo *q)
{
  q->head = q->tail = q->count = 0;
}

int enqueue_int_fifo(IntFifo *q, int value)
{
  if (q->count >= MAX)
    return 0; // Full
  q->data[q->tail] = value;
  q->tail = (q->tail + 1) % MAX;
  q->count++;
  return 1;
}

int dequeue_int_fifo(IntFifo *q, int *out)
{
  if (q->count <= 0)
    return 0; // Empty
  *out = q->data[q->head];
  q->head = (q->head + 1) % MAX;
  q->count--;
  return 1;
}