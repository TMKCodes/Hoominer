#include "stratum.h"
#include "platform_compat.h"

StratumContext *init_stratum_context()
{
  StratumContext *ctx = malloc(sizeof(StratumContext));
  if (!ctx)
  {
    fprintf(stderr, "Failed to allocate StratumContext\n");
    exit(1);
  }
  ctx->cpu_device_count = 0;
  ctx->opencl_device_count = 0;
  ctx->opencl_resources = NULL;
  ctx->cuda_device_count = 0;
  ctx->cuda_resources = NULL;
  ctx->worker = NULL;
  ctx->sockfd = -1;
  ctx->ssl = NULL;
  ctx->ssl_ctx = NULL;
  ctx->ms = NULL;
  ctx->hd = NULL;
  ctx->running = true;
  ctx->current_stratum_index = 0;
  ctx->disable_cpu = 0;
  ctx->disable_gpu = 0;
  ctx->recv_thread_created = 0;
  init_int_fifo(&ctx->mining_submit_fifo);

  // Initialize OpenSSL only if needed later
  SSL_load_error_strings();
  OpenSSL_add_ssl_algorithms();

  return ctx;
}

int init_ssl_connection(StratumContext *ctx)
{
  // Create SSL context
  ctx->ssl_ctx = SSL_CTX_new(TLS_client_method());
  if (!ctx->ssl_ctx)
  {
    fprintf(stderr, "Failed to create SSL context\n");
    ERR_print_errors_fp(stderr);
    return -1;
  }

  // Set SSL options to disable deprecated protocols
  SSL_CTX_set_options(ctx->ssl_ctx, SSL_OP_NO_SSLv2 | SSL_OP_NO_SSLv3 | SSL_OP_NO_TLSv1 | SSL_OP_NO_TLSv1_1);

  // Create SSL object
  ctx->ssl = SSL_new(ctx->ssl_ctx);
  if (!ctx->ssl)
  {
    fprintf(stderr, "Failed to create SSL object\n");
    ERR_print_errors_fp(stderr);
    SSL_CTX_free(ctx->ssl_ctx);
    ctx->ssl_ctx = NULL;
    return -1;
  }

  // Bind SSL to socket
  if (SSL_set_fd(ctx->ssl, ctx->sockfd) != 1)
  {
    fprintf(stderr, "Failed to bind SSL to socket\n");
    ERR_print_errors_fp(stderr);
    SSL_free(ctx->ssl);
    ctx->ssl = NULL;
    SSL_CTX_free(ctx->ssl_ctx);
    ctx->ssl_ctx = NULL;
    return -1;
  }

  // Perform SSL handshake
  if (SSL_connect(ctx->ssl) != 1)
  {
    fprintf(stderr, "SSL handshake failed\n");
    ERR_print_errors_fp(stderr);
    SSL_free(ctx->ssl);
    ctx->ssl = NULL;
    SSL_CTX_free(ctx->ssl_ctx);
    ctx->ssl_ctx = NULL;
    return -1;
  }

  printf("SSL connection established\n");
  return 0;
}

int start_stratum_connection(StratumContext *ctx, HoominerConfig *config)
{
  winsock_init_once();
  if (config->stratum_urls_num > 0)
  {
    printf("Stratum URLs:\n");
    for (int j = 0; j < config->stratum_urls_num; j++)
    {
      struct StratumConfig *stratum = &config->stratum_urls[j];
      const char *protocol = stratum->ssl_enabled ? "stratum+ssl://" : "stratum+tcp://";
      printf("  %d: %s%s:%d\n", j + 1, protocol, stratum->pool_ip, stratum->pool_port);
    }
  }
  StratumConfig *stratumConfig = get_stratum(config, ctx->current_stratum_index);
  if (!stratumConfig)
  {
    fprintf(stderr, "No valid stratum configuration available.\n");
    return -1;
  }
  printf("Opening connection to %s, stratum at %d index\n", stratumConfig->pool_ip, ctx->current_stratum_index + 1);
  ctx->current_stratum_index++;
  ctx->sockfd = connect_to_stratum_server(stratumConfig->pool_ip, stratumConfig->pool_port);
  if (ctx->sockfd < 0)
  {
    printf("Failed to connect to stratum server %s.\n", stratumConfig->pool_ip);
    return -1;
  }

  // Initialize SSL if enabled
  if (stratumConfig->ssl_enabled)
  {
    if (init_ssl_connection(ctx) < 0)
    {
      printf("Failed to initialize SSL connection.\n");
      socket_close_portable(ctx->sockfd);
      ctx->sockfd = -1;
      return -1;
    }
  }

  if (stratum_subscribe(ctx->sockfd, ctx, stratumConfig->ssl_enabled ? ctx->ssl : NULL) < 0 ||
      stratum_authenticate(ctx->sockfd, config->username, config->password, stratumConfig->ssl_enabled ? ctx->ssl : NULL) < 0)
  {
    printf("Stratum initialization failed.\n");
    if (ctx->ssl)
    {
      SSL_shutdown(ctx->ssl);
      SSL_free(ctx->ssl);
      ctx->ssl = NULL;
      SSL_CTX_free(ctx->ssl_ctx);
      ctx->ssl_ctx = NULL;
    }
    socket_close_portable(ctx->sockfd);
    ctx->sockfd = -1;
    return -1;
  }

  if (pthread_create(&ctx->recv_thread, NULL, stratum_receive_thread, ctx) != 0)
  {
    printf("Failed to create display or receive threads.\n");
    if (ctx->ssl)
    {
      SSL_shutdown(ctx->ssl);
      SSL_free(ctx->ssl);
      ctx->ssl = NULL;
      SSL_CTX_free(ctx->ssl_ctx);
      ctx->ssl_ctx = NULL;
    }
    socket_close_portable(ctx->sockfd);
    ctx->sockfd = -1;
    return -1;
  }
  else
  {
    ctx->recv_thread_created = 1;
  }
  if (pthread_create(&ctx->hd->display_thread, NULL, hashrate_display_thread, ctx) != 0)
  {
    printf("Failed to create display or receive threads.\n");
    if (ctx->ssl)
    {
      SSL_shutdown(ctx->ssl);
      SSL_free(ctx->ssl);
      ctx->ssl = NULL;
      SSL_CTX_free(ctx->ssl_ctx);
      ctx->ssl_ctx = NULL;
    }
    close(ctx->sockfd);
    ctx->sockfd = -1;
    return -1;
  }
  else
  {
    ctx->hd->display_thread_created = 1;
  }

  if (start_mining_threads(ctx, ctx->ms) != 0)
  {
    printf("Failed to start mining threads.\n");
    pthread_cancel(ctx->hd->display_thread);
    pthread_join(ctx->hd->display_thread, NULL);
    pthread_cancel(ctx->recv_thread);
    pthread_join(ctx->recv_thread, NULL);
    if (ctx->ssl)
    {
      SSL_shutdown(ctx->ssl);
      SSL_free(ctx->ssl);
      ctx->ssl = NULL;
      SSL_CTX_free(ctx->ssl_ctx);
      ctx->ssl_ctx = NULL;
    }
    close(ctx->sockfd);
    ctx->sockfd = -1;
    return -1;
  }

  return 0;
}

void process_stratum_message(json_object *message, StratumContext *ctx, MiningState *ms)
{
  if (!message)
  {
    printf("process_stratum_message: Null message received\n");
    return;
  }
  // printf("%s\n", json_object_to_json_string_ext(message, JSON_C_TO_STRING_PRETTY));

  json_object *method_obj;
  if (json_object_object_get_ex(message, "method", &method_obj) && json_object_is_type(method_obj, json_type_string))
  {
    const char *method_str = json_object_get_string(method_obj);
    if (!strcmp(method_str, "mining.set_difficulty"))
    {
      // {
      //   "id":null,
      //   "jsonrpc":"2.0",
      //   "method":"mining.set_difficulty",
      //   "params":[
      //     0.0035335689045936395
      //   ]
      // }
      json_object *params;
      if (json_object_object_get_ex(message, "params", &params) && json_object_is_type(params, json_type_array))
      {
        json_object *diff = json_object_array_get_idx(params, 0);
        if (json_object_is_type(diff, json_type_double) || json_object_is_type(diff, json_type_int))
        {
          pthread_mutex_lock(&ms->target_mutex);
          if (ms->global_target)
            free(ms->global_target);
          double difficulty = json_object_get_double(diff);
          ms->global_target = target_from_pool_difficulty(difficulty, DOMAIN_HASH_SIZE);
          if (ctx->config->debug == 1)
            printf("Received new difficulty %f\n", difficulty);
          pthread_mutex_unlock(&ms->target_mutex);
        }
      }
    }
    else if (!strcmp(method_str, "set_extranonce"))
    {
      // {
      //   "id":null,
      //   "jsonrpc":"2.0",
      //   "method":"set_extranonce",
      //   "params":[
      //     "00"
      //   ]
      // }
      json_object *params;
      if (!json_object_object_get_ex(message, "params", &params) || !json_object_is_type(params, json_type_array))
      {
        printf("mining.notify: params missing or not an array\n");
        return;
      }
      json_object *extranonce_param = json_object_array_get_idx(params, 0);

      if (json_object_is_type(extranonce_param, json_type_string))
      {
        const char *new_extranonce = json_object_get_string(extranonce_param);
        // Free old extranonce if any
        if (ms->extranonce)
        {
          free(ms->extranonce);
          ms->extranonce = NULL;
        }
        // Duplicate the string since JSON object will be freed
        if (new_extranonce)
        {
          ms->extranonce = strdup(new_extranonce);
        }
      }
    }
    else if (!strcmp(method_str, "mining.notify"))
    {
      // {
      //   "id":1,
      //   "jsonrpc":"2.0",
      //   "method":"mining.notify",
      //   "params":[
      //     "1",
      //     [
      //       9680649812803242576,
      //       5479210451378731168,
      //       9916984324433298320,
      //       15931787360748167486
      //     ],
      //     1754036484859
      //   ]
      // }

      json_object *params;
      if (!json_object_object_get_ex(message, "params", &params) || !json_object_is_type(params, json_type_array))
      {
        printf("mining.notify: params missing or not an array\n");
        return;
      }

      json_object *job_id = json_object_array_get_idx(params, 0);
      json_object *header_item = json_object_array_get_idx(params, 1);
      json_object *time_param = json_object_array_get_idx(params, 2);

      if (!json_object_is_type(job_id, json_type_string) || !header_item || !time_param ||
          (!json_object_is_type(time_param, json_type_int) && !json_object_is_type(time_param, json_type_double)))
      {
        printf("mining.notify: Invalid parameters\n");
        return;
      }

      uint8_t header[DOMAIN_HASH_SIZE] = {0};
      uint64_t timestamp_int;

      if (json_object_is_type(time_param, json_type_int))
        timestamp_int = json_object_get_uint64(time_param);
      else
        timestamp_int = (uint64_t)json_object_get_double(time_param);

      if (json_object_is_type(header_item, json_type_array))
      {
        if (json_object_array_length(header_item) != 4)
        {
          printf("Invalid header array size: %zu\n", json_object_array_length(header_item));
          return;
        }
        uint64_t hash_elements[4] = {0};
        for (int i = 0; i < 4; i++)
        {
          json_object *item = json_object_array_get_idx(header_item, i);
          if (json_object_is_type(item, json_type_int))
          {
            hash_elements[i] = json_object_get_uint64(item);
          }
          else if (json_object_is_type(item, json_type_string))
          {
            if (sscanf(json_object_get_string(item), "%" SCNx64, &hash_elements[i]) != 1)
            {
              return;
            }
          }
          else
          {
            printf("Invalid header element at index %d\n", i);
            return;
          }
        }
        smallJobHeader(hash_elements, header);
      }
      else if (json_object_is_type(header_item, json_type_string))
      {
        const char *hex_str = json_object_get_string(header_item);
        if (strlen(hex_str) != 64)
        {
          printf("Invalid hex string length: %zu\n", strlen(hex_str));
          return;
        }
        if (hex_to_bytes(hex_str, header, DOMAIN_HASH_SIZE) != 0)
        {
          printf("Failed to parse hex header: %s\n", hex_str);
          return;
        }
        print_hex("Parsed Hex Header", header, DOMAIN_HASH_SIZE);
      }
      else
      {
        printf("Invalid header format in mining.notify\n");
        return;
      }

      pthread_mutex_lock(&ms->job_queue.queue_mutex);
      while (ms->job_queue.head != ms->job_queue.tail &&
             ms->job_queue.jobs[ms->job_queue.head].timestamp < (uint64_t)time(NULL) - JOB_MAX_AGE)
      {
        free(ms->job_queue.jobs[ms->job_queue.head].job_id);
        ms->job_queue.jobs[ms->job_queue.head].job_id = NULL;
        ms->job_queue.head = (ms->job_queue.head + 1) % JOB_QUEUE_SIZE;
      }

      int next_tail = (ms->job_queue.tail + 1) % JOB_QUEUE_SIZE;
      if (next_tail == ms->job_queue.head)
      {
        free(ms->job_queue.jobs[ms->job_queue.head].job_id);
        ms->job_queue.jobs[ms->job_queue.head].job_id = NULL;
        ms->job_queue.head = (ms->job_queue.head + 1) % JOB_QUEUE_SIZE;
      }

      QueuedJob *new_job = &ms->job_queue.jobs[ms->job_queue.tail];
      const char *jid = json_object_get_string(job_id);
      if (jid)
      {
        new_job->job_id = strdup(jid);
        if (new_job->job_id)
        {
          if (ctx->config->debug == 1)
            printf("Received new job %s\n", new_job->job_id);
          memcpy(new_job->header, header, DOMAIN_HASH_SIZE);
          new_job->timestamp = timestamp_int;
          new_job->running = 1;
          new_job->completed = 0;
          generateHoohashMatrix(header, new_job->matrix);
          ms->job_queue.tail = next_tail;
          ms->new_job_available = 1;
          pthread_cond_broadcast(&ms->job_queue.queue_cond);
        }
        else
        {
          printf("Failed to allocate memory for job ID\n");
        }
      }
      else
      {
        printf("Job ID string is NULL\n");
      }
      pthread_mutex_unlock(&ms->job_queue.queue_mutex);
    }
  }
  else
  {
    json_object *result;
    json_object *error;
    int devices = ctx->cpu_device_count + ctx->opencl_device_count + ctx->cuda_device_count;
    if (devices > 0)
    {
      int device_index;
      int dequeue_result = dequeue_int_fifo(&ctx->mining_submit_fifo, &device_index);
      if (dequeue_result == 1)
      {
        if (ctx->config->debug == 1)
        {
          printf("device index %d\n", device_index);
          printf("deivces %d\n", devices);
        }
        if (devices > device_index && device_index >= 0)
        {
          ReportingDevice *device = ctx->hd->devices[device_index];
          if (json_object_object_get_ex(message, "error", &error))
          {
            if (!json_object_is_type(error, json_type_null))
            {
              if (json_object_is_type(error, json_type_array))
              {
                json_object *code = json_object_array_get_idx(error, 0);
                int err_code = json_object_get_int(code);
                if (err_code == 21)
                {
                  pthread_mutex_lock(&ms->job_queue.queue_mutex);
                  device->stales++;
                  pthread_mutex_unlock(&ms->job_queue.queue_mutex);
                }
                else if (err_code == 20)
                {
                  pthread_mutex_lock(&ms->job_queue.queue_mutex);
                  device->stales++;
                  pthread_mutex_unlock(&ms->job_queue.queue_mutex);
                }
                else
                {
                  const char *result_str = json_object_to_json_string(message);
                  printf("Error: %s\n", result_str);
                  pthread_mutex_lock(&ms->job_queue.queue_mutex);
                  device->rejected++;
                  pthread_mutex_unlock(&ms->job_queue.queue_mutex);
                }
              }
            }
          }
          if (json_object_object_get_ex(message, "result", &result))
          {
            if (json_object_is_type(result, json_type_boolean))
            {
              pthread_mutex_lock(&ms->job_queue.queue_mutex);
              if (json_object_get_boolean(result))
                device->accepted++;
              else
                device->rejected++;
              pthread_mutex_unlock(&ms->job_queue.queue_mutex);
            }
          }
        }
      }
    }
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
    struct timeval tv = {0, 100000}; // 100ms timeout
    FD_ZERO(&fds);
    FD_SET(ctx->sockfd, &fds);

    if (select(ctx->sockfd + 1, &fds, NULL, NULL, &tv) <= 0)
      continue;

    int bytes;
    if (ctx->ssl)
    {
      bytes = SSL_read(ctx->ssl, buffer, BUFFER_SIZE - 1);
      if (bytes <= 0)
      {
        fprintf(stderr, "stratum_receive_thread: SSL connection lost or error (%d)\n", bytes);
        ERR_print_errors_fp(stderr);
        ctx->running = 0;
        break;
      }
    }
    else
    {
      bytes = recv(ctx->sockfd, buffer, BUFFER_SIZE - 1, 0);
      if (bytes <= 0)
      {
        printf("stratum_receive_thread: Connection lost or error (%d)\n", bytes);
        ctx->running = 0;
        break;
      }
    }
    if (bytes >= 0)
    {
      buffer[bytes] = '\0';
      if (json_len + bytes >= sizeof(json_buffer))
      {
        printf("stratum_receive_thread: Buffer overflow, processing partial\n");
        char *start = json_buffer;
        char *end;
        char *partial_end = json_buffer + json_len;
        while ((end = strchr(start, '\n')) && end < partial_end)
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
        memmove(json_buffer, start, json_len);
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
  }

  // Cleanup SSL if enabled
  if (ctx->ssl)
  {
    SSL_shutdown(ctx->ssl);
    SSL_free(ctx->ssl);
    ctx->ssl = NULL;
    SSL_CTX_free(ctx->ssl_ctx);
    ctx->ssl_ctx = NULL;
  }
  return NULL;
}

int stratum_subscribe(int sockfd, StratumContext *ctx, SSL *ssl)
{
  json_object *req = json_object_new_object();
  json_object_object_add(req, "id", json_object_new_int(1));
  json_object_object_add(req, "method", json_object_new_string("mining.subscribe"));
  json_object *params = json_object_new_array();
  char version_string[128];
  snprintf(version_string, sizeof(version_string), "Hoominer/%s", ctx->version);
  json_object_array_add(params, json_object_new_string(version_string));
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

  strncpy(msg_with_newline, msg, len);
  msg_with_newline[len] = '\n';
  msg_with_newline[len + 1] = '\0';

  int ret;
  if (ssl)
  {
    ret = SSL_write(ssl, msg_with_newline, len + 1);
  }
  else
  {
    ret = send(sockfd, msg_with_newline, len + 1, 0);
  }
  if (ret != 0)
  {
    printf("Stratum subscription message sent:\n %s", msg_with_newline);
  }
  free(msg_with_newline);
  json_object_put(req);
  return ret < 0 ? -1 : 0;
}

int stratum_authenticate(int sockfd, const char *username, const char *password, SSL *ssl)
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

  strncpy(msg_with_newline, msg, len);
  msg_with_newline[len] = '\n';
  msg_with_newline[len + 1] = '\0';

  int ret;
  if (ssl)
  {
    ret = SSL_write(ssl, msg_with_newline, len + 1);
  }
  else
  {
    ret = send(sockfd, msg_with_newline, len + 1, 0);
  }
  if (ret != 0)
  {
    printf("Stratum authenticate message sent:\n%s", msg_with_newline);
  }
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

  // Try both IPv4 and IPv6
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_ADDRCONFIG;

  printf("Attempting to resolve %s:%d...\n", hostname, port);
  int gai_result = getaddrinfo(hostname, port_str, &hints, &res);
  if (gai_result != 0)
  {
    fprintf(stderr, "getaddrinfo error for %s: %s\n", hostname, gai_strerror(gai_result));

    // Provide more specific error messages
    switch (gai_result)
    {
    case EAI_NONAME:
      fprintf(stderr, "Hostname '%s' could not be resolved. Check if the hostname is correct and DNS is working.\n", hostname);
      break;
    case EAI_AGAIN:
      fprintf(stderr, "Temporary failure in name resolution. Try again later.\n");
      break;
    case EAI_FAIL:
      fprintf(stderr, "Non-recoverable failure in name resolution.\n");
      break;
    default:
      fprintf(stderr, "Unknown getaddrinfo error: %d\n", gai_result);
      break;
    }
    return -1;
  }

  for (p = res; p != NULL; p = p->ai_next)
  {
    sockfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (sockfd < 0)
    {
      perror("socket creation failed");
      continue;
    }

    int flag = 1;
    setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(int));

    printf("Attempting to connect to %s:%d (family: %s)...\n",
           hostname, port, (p->ai_family == AF_INET) ? "IPv4" : "IPv6");

    if (connect(sockfd, p->ai_addr, p->ai_addrlen) == 0)
    {
      char ip_str[INET6_ADDRSTRLEN];
      if (p->ai_family == AF_INET)
      {
        struct sockaddr_in *ipv4 = (struct sockaddr_in *)p->ai_addr;
        inet_ntop(AF_INET, &(ipv4->sin_addr), ip_str, INET_ADDRSTRLEN);
      }
      else
      {
        struct sockaddr_in6 *ipv6 = (struct sockaddr_in6 *)p->ai_addr;
        inet_ntop(AF_INET6, &(ipv6->sin6_addr), ip_str, INET6_ADDRSTRLEN);
      }
      printf("Connected to %s (resolved IP: %s)\n", hostname, ip_str);
      break;
    }
    else
    {
      perror("connect failed");
    }
    socket_close_portable(sockfd);
  }

  freeaddrinfo(res);
  if (p == NULL)
  {
    fprintf(stderr, "Failed to connect to %s:%d - all connection attempts failed\n", hostname, port);
    fprintf(stderr, "Possible causes:\n");
    fprintf(stderr, "  - Host is unreachable or down\n");
    fprintf(stderr, "  - Port is not open or blocked by firewall\n");
    fprintf(stderr, "  - Network connectivity issues\n");
    fprintf(stderr, "  - DNS resolution problems (if using hostname)\n");
    return -1;
  }
  return sockfd;
}

void free_stratum_context(StratumContext *ctx)
{
  if (!ctx)
    return;

  if (ctx->ssl)
  {
    SSL_shutdown(ctx->ssl);
    SSL_free(ctx->ssl);
    ctx->ssl = NULL;
  }
  if (ctx->ssl_ctx)
  {
    SSL_CTX_free(ctx->ssl_ctx);
    ctx->ssl_ctx = NULL;
  }
  if (ctx->sockfd >= 0)
  {
    socket_close_portable(ctx->sockfd);
    ctx->sockfd = -1;
  }
  // Note: ctx->config is freed by cleanup() in hoominer.c
  free(ctx);
}