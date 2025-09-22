@echo off

echo Installing dependencies with vcpkg...
C:\vcpkg\vcpkg install openssl:x64-windows json-c:x64-windows libmicrohttpd:x64-windows opencl:x64-windows blake3:x64-windows

echo Removing existing build folder...
if exist build (
    rmdir /s /q build
)

echo Creating new build folder...
mkdir build

echo Configuring CMake project...
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake

echo Building project...
cmake --build build --config Release

echo Creating releases/hoominer folder...
if not exist releases\hoominer (
    mkdir releases\hoominer
)

echo Moving build/Release files to releases/hoominer...
move build\Release\*.* releases\hoominer\

echo Done.
pause