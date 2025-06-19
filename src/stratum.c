#define _POSIX_C_SOURCE 200112L
#include "stratum.h"

StratumContext *init_stratum_context()
{
  StratumContext *ctx = malloc(sizeof(StratumContext));
  if (!ctx)
  {
    fprintf(stderr, "Failed to allocate StratumContext\n");
    exit(1);
  }
  ctx->disable_cpu = 0;
  ctx->disable_gpu = 0;
  ctx->opencl_device_count = 0;
  ctx->opencl_resources = NULL;
  ctx->worker = NULL;
  ctx->sockfd = -1;
  ctx->ms = NULL;
  ctx->hd = NULL;
  ctx->running = true;
  return ctx;
}

void cleanup_stratum_context(StratumContext *ctx)
{
  if (ctx)
  {
    free(ctx);
  }
}

void *stratum_receive_thread(void *arg)
{
  StratumContext *ctx = (StratumContext *)arg;
  char buffer[BUFFER_SIZE];
  char json_buffer[BUFFER_SIZE * 2] = {0};
  size_t json_len = 0;

  while (ctx->running)
  {
    fd_set fds;
    struct timeval tv = {1, 0};
    FD_ZERO(&fds);
    FD_SET(ctx->sockfd, &fds);

    if (select(ctx->sockfd + 1, &fds, NULL, NULL, &tv) <= 0)
      continue;

    int bytes = recv(ctx->sockfd, buffer, BUFFER_SIZE - 1, 0);
    if (bytes <= 0)
    {
      printf("stratum_receive_thread: Connection lost or error (%d)\n", bytes);
      ctx->running = 0;
      break;
    }

    buffer[bytes] = '\0';
    if (json_len + bytes >= sizeof(json_buffer))
    {
      printf("stratum_receive_thread: Buffer overflow, resetting\n");
      json_len = 0;
      continue;
    }

    memcpy(json_buffer + json_len, buffer, bytes);
    json_len += bytes;

    char *start = json_buffer;
    char *end;
    while ((end = strchr(start, '\n')))
    {
      *end = '\0';
      json_object *msg = json_tokener_parse(start);
      if (msg)
      {
        process_stratum_message(msg, ctx, ctx->ms);
        json_object_put(msg);
      }
      else
        printf("stratum_receive_thread: Failed to parse JSON: %s\n", start);
      json_len -= (end - start + 1);
      start = end + 1;
    }
    if (json_len > 0 && start != NULL)
    {
      memmove(json_buffer, start, json_len);
    }
  }
  return NULL;
}

int stratum_subscribe(int sockfd)
{
  json_object *req = json_object_new_object();
  json_object_object_add(req, "id", json_object_new_int(1));
  json_object_object_add(req, "method", json_object_new_string("mining.subscribe"));
  json_object *params = json_object_new_array();
  json_object_array_add(params, json_object_new_string("Hoominer/0.0.0"));
  json_object_object_add(req, "params", params);

  const char *msg = json_object_to_json_string_ext(req, JSON_C_TO_STRING_PLAIN);
  if (!msg)
  {
    json_object_put(req);
    return -1;
  }
  size_t len = strlen(msg);
  char *msg_with_newline = malloc(len + 2);
  if (!msg_with_newline)
  {
    json_object_put(req);
    return -1;
  }

  strcpy(msg_with_newline, msg);
  msg_with_newline[len] = '\n';
  msg_with_newline[len + 1] = '\0';

  int ret = send(sockfd, msg_with_newline, len + 1, 0);
  free(msg_with_newline);
  json_object_put(req);
  return ret < 0 ? -1 : 0;
}

int stratum_authenticate(int sockfd, const char *username, const char *password)
{
  json_object *req = json_object_new_object();
  json_object_object_add(req, "id", json_object_new_int(1));
  json_object_object_add(req, "method", json_object_new_string("mining.authorize"));
  json_object *params = json_object_new_array();
  json_object_array_add(params, json_object_new_string(username));
  json_object_array_add(params, json_object_new_string(password));
  json_object_object_add(req, "params", params);

  const char *msg = json_object_to_json_string_ext(req, JSON_C_TO_STRING_PLAIN);
  if (!msg)
  {
    json_object_put(req);
    return -1;
  }
  size_t len = strlen(msg);
  char *msg_with_newline = malloc(len + 2);
  if (!msg_with_newline)
  {
    json_object_put(req);
    return -1;
  }

  strcpy(msg_with_newline, msg);
  msg_with_newline[len] = '\n';
  msg_with_newline[len + 1] = '\0';

  int ret = send(sockfd, msg_with_newline, len + 1, 0);
  free(msg_with_newline);
  json_object_put(req);
  return ret < 0 ? -1 : 0;
}

int connect_to_stratum_server(const char *hostname, int port)
{
  struct addrinfo hints = {0}, *res, *p;
  int sockfd;
  char port_str[6];
  snprintf(port_str, sizeof(port_str), "%d", port);

  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;

  if (getaddrinfo(hostname, port_str, &hints, &res) != 0)
  {
    perror("getaddrinfo error");
    return -1;
  }

  for (p = res; p != NULL; p = p->ai_next)
  {
    sockfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (sockfd < 0)
      continue;

    if (connect(sockfd, p->ai_addr, p->ai_addrlen) == 0)
    {
      char ip_str[INET_ADDRSTRLEN];
      struct sockaddr_in *ipv4 = (struct sockaddr_in *)p->ai_addr;
      inet_ntop(AF_INET, &(ipv4->sin_addr), ip_str, INET_ADDRSTRLEN);
      printf("Connected to %s (resolved IP: %s)\n", hostname, ip_str);
      break;
    }
    close(sockfd);
  }

  freeaddrinfo(res);
  if (p == NULL)
  {
    perror("Failed to connect");
    return -1;
  }
  return sockfd;
}