#ifndef REPORTING_H
#define REPORTING_H
#include <pthread.h>
#include <stdint.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

typedef struct HashrateDisplay HashrateDisplay;
HashrateDisplay *init_hashrate_display();
void *hashrate_display_thread(void *arg);

#endif