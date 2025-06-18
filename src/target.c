#include "target.h"

double difficulty_from_target(uint8_t *target, size_t len)
{
  mpz_t target_val, max_val, quotient;
  mpz_init(target_val);
  mpz_init(max_val);
  mpz_init(quotient);

  mpz_import(target_val, len, 1, sizeof(uint8_t), 0, 0, target);
  mpz_set_str(max_val, "00000000FFFF0000000000000000000000000000000000000000000000000000", 16);

  if (mpz_cmp_ui(target_val, 0) == 0)
  {
    mpz_clear(target_val);
    mpz_clear(max_val);
    mpz_clear(quotient);
    return 0.0;
  }

  mpz_tdiv_q(quotient, max_val, target_val);
  double difficulty = mpz_get_d(quotient);

  mpz_clear(target_val);
  mpz_clear(max_val);
  mpz_clear(quotient);
  return difficulty;
}

uint8_t *target_from_pool_difficulty(double difficulty, size_t len)
{
  if (difficulty <= 0)
    return NULL;

  mpz_t max_target, target;
  mpf_t diff_mpf, max_mpf;

  mpz_init(max_target);
  mpz_init(target);
  mpf_init2(diff_mpf, 512);
  mpf_init2(max_mpf, 512);

  mpz_set_str(max_target, "00000000FFFF0000000000000000000000000000000000000000000000000000", 16);
  mpf_set_d(diff_mpf, difficulty);
  mpf_set_z(max_mpf, max_target);
  mpf_div(diff_mpf, max_mpf, diff_mpf);
  mpf_floor(diff_mpf, diff_mpf);
  mpz_set_f(target, diff_mpf);

  uint8_t *target_bytes = calloc(len, sizeof(uint8_t));
  if (!target_bytes)
    goto cleanup;

  size_t count;
  mpz_export(target_bytes, &count, 1, sizeof(uint8_t), 0, 0, target);
  if (count < len)
  {
    memmove(target_bytes + (len - count), target_bytes, count);
    memset(target_bytes, 0, len - count);
  }
  else if (count > len)
  {
    memmove(target_bytes, target_bytes + (count - len), len);
  }

  // printf("Computed Target: 0x");
  // for (int i = 0; i < len; i++)
  //   printf("%02x", target_bytes[i]);
  // printf("\nComputed Difficulty: %.6f\n", difficulty_from_target(target_bytes));

cleanup:
  mpz_clear(max_target);
  mpz_clear(target);
  mpf_clear(diff_mpf);
  mpf_clear(max_mpf);
  return target_bytes;
}

int compare_target(uint8_t *hash, uint8_t *target, size_t len)
{
  uint8_t reversed_hash[len];
  for (size_t i = 0; i < len; i++)
  {
    reversed_hash[i] = hash[len - 1 - i];
  }
  for (size_t i = 0; i < len; i++)
  {
    if (reversed_hash[i] > target[i])
      return 1;
    if (reversed_hash[i] < target[i])
      return -1;
  }
  return 0;
}