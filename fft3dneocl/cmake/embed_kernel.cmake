# Embeds a .cl kernel source into a C++ header with raw string literals.
# Usage: cmake -DCL_SRC=<file.cl> -DOUT=<header.hpp> -P embed_kernel.cmake
# MSVC limits one string literal to 16380 bytes (C2026), so the source is
# split into 8000-byte chunks that the compiler concatenates.
file(READ "${CL_SRC}" SRC)
string(FIND "${SRC}" "\${" BAD1 REVERSE)
string(FIND "${SRC}" ")FFT3DCL\"" BAD2 REVERSE)
if(BAD1 GREATER -1 OR BAD2 GREATER -1)
  message(FATAL_ERROR "kernel source contains a sequence that breaks embedding")
endif()

string(LENGTH "${SRC}" LEN)
set(CHUNKS "")
set(OFF 0)
while(OFF LESS LEN)
  string(SUBSTRING "${SRC}" ${OFF} 8000 CHUNK)
  string(APPEND CHUNKS "R\"FFT3DCL(${CHUNK})FFT3DCL\"\n")
  math(EXPR OFF "${OFF} + 8000")
endwhile()
file(WRITE "${OUT}"
     "#pragma once\nstatic const char FFT3D_CL_SRC[] =\n${CHUNKS};\n")
