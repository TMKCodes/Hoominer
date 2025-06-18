#ifndef TARGET_H
#define TARGET_H
#include <stdint.h>
#include <gmp.h>
#include <stdlib.h>
#include <string.h>

double difficulty_from_target(uint8_t *target, size_t len);
uint8_t *target_from_pool_difficulty(double difficulty, size_t len);
int compare_target(uint8_t *hash, uint8_t *target, size_t len);

#endif