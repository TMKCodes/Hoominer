#include "api.h"

// Callback for handling HTTP requests
static enum MHD_Result request_handler(void *cls, struct MHD_Connection *connection,
                                       const char *url, const char *method,
                                       const char *version, const char *upload_data,
                                       size_t *upload_data_size, void **con_cls)
{
  struct StratumContext *ctx = (struct StratumContext *)cls;

  struct MHD_Response *response;
  enum MHD_Result ret;
  char *json_response;

  // Handle GET request
  if (strcmp(method, "GET") == 0)
  {
    if (strcmp(url, "/api/hello") == 0)
    {
      // Include server name and request count in response
      json_response = malloc(128);
      snprintf(json_response, 128, "{\"message\": \"Hello from Hoominer\"}");
      response = MHD_create_response_from_buffer(strlen(json_response),
                                                 json_response,
                                                 MHD_RESPMEM_MUST_FREE);
      MHD_add_response_header(response, "Content-Type", "application/json");
      ret = MHD_queue_response(connection, MHD_HTTP_OK, response);
      MHD_destroy_response(response);
      return ret;
    }
    if (strcmp(url, "/gpu") == 0)
    {
      struct json_object *root = json_object_new_object();

      // busids array
      struct json_object *busid_array = json_object_new_array();
      if (ctx->disable_cpu == false)
      {
        json_object_array_add(busid_array, json_object_new_string("cpu"));
      }
      if (ctx->disable_gpu == false)
      {
        for (unsigned int i = 0; i < ctx->opencl_device_count; i++)
        {
          json_object_array_add(busid_array, json_object_new_int(ctx->opencl_resources[i].pci_bus_id));
        }
        for (unsigned int i = 0; i < ctx->cuda_device_count; i++)
        {
          json_object_array_add(busid_array, json_object_new_int(ctx->cuda_resources[i].pci_bus_id));
        }
      }
      json_object_object_add(root, "busid", busid_array);

      struct json_object *hash_array = json_object_new_array();
      for (uint32_t i = 0; i < ctx->hd->device_count; i++)
      {
        json_object_array_add(hash_array, json_object_new_int(ctx->hd->devices[i]->hashrate));
      }
      json_object_object_add(root, "hash", hash_array);
      struct json_object *units_string = json_object_new_string("hs");
      json_object_object_add(root, "units", units_string);

      struct json_object *air_object = json_object_new_array();

      json_object_object_add(root, "air", air_object);
      int all_accepted = 0;
      for (uint32_t i = 0; i < ctx->hd->device_count; i++)
      {
        all_accepted += ctx->hd->devices[i]->accepted;
      }

      int all_rejected = 0;
      for (uint32_t i = 0; i < ctx->hd->device_count; i++)
      {
        all_rejected += ctx->hd->devices[i]->rejected;
      }
      json_object_array_add(air_object, json_object_new_int(all_accepted));
      json_object_array_add(air_object, json_object_new_int(0));
      json_object_array_add(air_object, json_object_new_int(all_rejected));

      struct json_object *shares_object = json_object_new_object();
      struct json_object *accepted_shares = json_object_new_array();
      struct json_object *rejected_shares = json_object_new_array();
      struct json_object *invalid_shares = json_object_new_array();

      for (uint32_t i = 0; i < ctx->hd->device_count; i++)
      {
        json_object_array_add(accepted_shares, json_object_new_int(ctx->hd->devices[i]->accepted));
      }

      for (uint32_t i = 0; i < ctx->hd->device_count; i++)
      {
        json_object_array_add(rejected_shares, json_object_new_int(ctx->hd->devices[i]->stales + ctx->hd->devices[i]->rejected));
      }

      for (uint32_t i = 0; i < ctx->hd->device_count; i++)
      {
        json_object_array_add(invalid_shares, json_object_new_int(0)); // Add tracking maybe later.
      }

      json_object_object_add(shares_object, "accepted", accepted_shares);
      json_object_object_add(shares_object, "rejected", rejected_shares);
      json_object_object_add(shares_object, "invalid", invalid_shares);
      json_object_object_add(root, "shares", shares_object);

      struct json_object *name_string = json_object_new_string("Hoominer");
      json_object_object_add(root, "miner_name", name_string);

      struct json_object *version_string = json_object_new_string(ctx->version);
      json_object_object_add(root, "miner_version", version_string);

      const char *json_response = json_object_to_json_string(root);
      response = MHD_create_response_from_buffer(strlen(json_response), json_response, MHD_RESPMEM_MUST_FREE);
      MHD_add_response_header(response, "Content-Type", "application/json");
      ret = MHD_queue_response(connection, MHD_HTTP_OK, response);
      MHD_destroy_response(response);
      return ret;
    }

    // Handle unknown endpoints
    json_response = "{\"error\": \"Not found\"}";
    response = MHD_create_response_from_buffer(strlen(json_response),
                                               json_response,
                                               MHD_RESPMEM_MUST_COPY);
    MHD_add_response_header(response, "Content-Type", "application/json");
    ret = MHD_queue_response(connection, MHD_HTTP_NOT_FOUND, response);
    MHD_destroy_response(response);
    return ret;
  }

  // Method not allowed
  json_response = "{\"error\": \"Method not allowed\"}";
  response = MHD_create_response_from_buffer(strlen(json_response),
                                             json_response,
                                             MHD_RESPMEM_MUST_COPY);
  MHD_add_response_header(response, "Content-Type", "application/json");
  ret = MHD_queue_response(connection, MHD_HTTP_METHOD_NOT_ALLOWED, response);
  MHD_destroy_response(response);
  return ret;
}

struct MHD_Daemon *start_api(struct StratumContext *ctx)
{
  struct MHD_Daemon *daemon;
  // Pass the ctx structure to MHD_start_daemon via cls
  daemon = MHD_start_daemon(MHD_USE_INTERNAL_POLLING_THREAD, 8042, NULL, NULL,
                            &request_handler, ctx, MHD_OPTION_END);
  if (NULL == daemon)
  {
    printf("Failed to start API\n");
    return NULL;
  }

  printf("API running on port 8042.\n");
  return daemon;
}

int stop_api(struct MHD_Daemon *daemon)
{
  MHD_stop_daemon(daemon);
  printf("API shutdown\n");
  return 0;
}