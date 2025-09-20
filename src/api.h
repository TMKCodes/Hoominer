#ifndef API_H
#define API_H

#include <microhttpd.h>
#include <json-c/json.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "stratum.h"

struct PostData
{
  char *data;
  size_t size;
};

struct MHD_Daemon *start_api(struct StratumContext *ctx);
int stop_api(struct MHD_Daemon *daemon);
static enum MHD_Result request_handler(void *cls, struct MHD_Connection *connection,
  const char *url, const char *method,
  const char *version, const char *upload_data,
  size_t *upload_data_size, void **con_cls);

#endif