#pragma once

// cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
// cmake --build build --config Release

#ifdef _WIN32
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>

static inline void sleep_ms(unsigned int ms)
{
    Sleep(ms);
}

static inline void socket_close_portable(int s)
{
    closesocket(s);
}

static inline void winsock_init_once(void)
{
    static int inited = 0;
    if (!inited)
    {
        WSADATA wsaData;
        WSAStartup(MAKEWORD(2, 2), &wsaData);
        inited = 1;
    }
}

static inline int malloc_trim(size_t pad)
{
    (void)pad;
    return 1; // Always succeed, no-op on Windows
}

static inline void get_exe_dir(char *buf, size_t buflen)
{
    DWORD n = GetModuleFileNameA(NULL, buf, (DWORD)buflen);
    if (n > 0 && n < buflen)
    {
        for (int i = (int)n - 1; i >= 0; --i)
        {
            if (buf[i] == '\\' || buf[i] == '/')
            {
                buf[i] = 0;
                break;
            }
        }
    }
    else if (buflen)
    {
        buf[0] = 0;
    }
}

#else
#include <unistd.h>
#include <time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <malloc.h>

static inline void sleep_ms(unsigned int ms)
{
    struct timespec ts = {ms / 1000, (ms % 1000) * 1000000};
    nanosleep(&ts, NULL);
}

static inline void socket_close_portable(int s) { close(s); }

static inline void winsock_init_once(void) { (void)0; }

#endif
