#include "reporting.h"

HashrateDisplay *init_hashrate_display()
{
  HashrateDisplay *hd = malloc(sizeof(HashrateDisplay));
  if (!hd)
    return NULL;

  hd->nonces_processed = 0;
  hd->accepted = 0;
  hd->rejected = 0;

  // Initialize the mutex, return NULL on failure
  if (pthread_mutex_init(&hd->hashrate_mutex, NULL) != 0)
  {
    free(hd);
    return NULL;
  }

  hd->running = 1;

  return hd;
}

void cleanup_hashrate_display(StratumContext *ctx)
{
  if (ctx && ctx->hd)
  {
    pthread_mutex_destroy(&ctx->hd->hashrate_mutex);
    free(ctx->hd);
    ctx->hd = NULL; // Prevent double-free
  }
}

void *hashrate_display_thread(void *arg)
{
  StratumContext *ctx = (StratumContext *)arg;
  HashrateDisplay *hd = ctx->hd;
  int seconds = 5;
  while (ctx->running)
  {
    sleep(seconds);
    pthread_mutex_lock(&hd->hashrate_mutex);
    double hashrate = hd->nonces_processed / seconds;
    hd->nonces_processed = 0;
    pthread_mutex_unlock(&hd->hashrate_mutex);

    time_t t = time(NULL);
    struct tm tm;
    localtime_r(&t, &tm);
    char time_str[9];
    strftime(time_str, sizeof(time_str), "%H:%M:%S", &tm);

    printf("[%-6s] ======================================================================================\n", time_str);
    printf("[%-6s] |[hoohash]\t\t\t\t| Accepted shares \t| Rejected shares \t|\n", time_str);
    printf("[%-6s] ", time_str);
    if (hashrate > 1000000000000)
    {
      printf("|Hashrate: %.2f TH/s\t\t|  ", hashrate);
    }
    else if (hashrate > 1000000000)
    {
      printf("|Hashrate: %.2f GH/s\t\t|  ", hashrate / 1000.0);
    }
    else if (hashrate > 1000000)
    {
      printf("|Hashrate: %.2f MH/s\t\t|  ", hashrate / 1000.0);
    }
    else if (hashrate > 1000)
    {
      printf("|Hashrate: %.2f KH/s\t\t|  ", hashrate / 1000.0);
    }
    else
    {
      printf("|Hashrate: %.2f H/s\t\t|  ", hashrate);
    }
    if (hd->accepted > 1000000000000)
    {
      printf("%ld \t| ", hd->accepted);
    }
    else if (hd->accepted > 1000)
    {
      printf("%ld \t\t| ", hd->accepted);
    }
    else
    {
      printf("%ld \t\t\t| ", hd->accepted);
    }
    if (hd->rejected > 1000000000000)
    {
      printf("%ld \t|\n", hd->rejected);
    }
    else if (hd->rejected > 1000)
    {
      printf("%ld \t\t|\n", hd->rejected);
    }
    else
    {
      printf("%ld \t\t\t|\n", hd->rejected);
    }

    printf("[%-6s] ======================================================================================\n", time_str);
  }
  return NULL;
}