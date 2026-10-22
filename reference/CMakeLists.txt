# Builds the ORIGINAL neo_FFT3D sources (the repository root, i.e. the
# upstream code this fork is based on) as a VapourSynth plugin. Used only
# to produce reference output for validation of the OpenCL port.
cmake_minimum_required(VERSION 3.15)
project(neo_fft3d_ref CXX)

set(ORIG "${CMAKE_CURRENT_SOURCE_DIR}/..")

set(SRC
  "${ORIG}/main.cpp"
  "${ORIG}/src/buffer.cpp"
  "${ORIG}/src/cpuid.cpp"
  "${ORIG}/src/helper.cpp"
  "${ORIG}/src/code_impl/Apply.cpp"
  "${ORIG}/src/code_impl/Apply_SSE2.cpp"
  "${ORIG}/src/code_impl/Apply_AVX.cpp"
  "${ORIG}/src/code_impl/Apply_AVX512.cpp"
  "${ORIG}/src/code_impl/Kalman.cpp"
  "${ORIG}/src/code_impl/Kalman_SSE2.cpp"
  "${ORIG}/src/code_impl/Kalman_AVX.cpp"
  "${ORIG}/src/code_impl/Kalman_AVX512.cpp"
  "${ORIG}/src/code_impl/Sharpen.cpp"
  "${ORIG}/src/code_impl/Sharpen_SSE2.cpp"
  "${ORIG}/src/code_impl/Sharpen_AVX.cpp"
  "${ORIG}/src/code_impl/Sharpen_AVX512.cpp"
)

add_library(neo-fft3d SHARED ${SRC})
set_property(TARGET neo-fft3d PROPERTY CXX_STANDARD 17)

# generate version.hpp for the build (the clone stays untouched)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/version.hpp" "#pragma once\n#define PLUGIN_VERSION \"ref\"\n")

target_include_directories(neo-fft3d PRIVATE
  "${ORIG}"
  "${ORIG}/include/avisynth"
  "${ORIG}/include/vapoursynth"
  "${ORIG}/include/dualsynth"
  "${CMAKE_CURRENT_BINARY_DIR}"
)

target_compile_definitions(neo-fft3d PRIVATE HAS_EXECUTION ENABLE_PAR)

if(MSVC)
  set_source_files_properties(
    "${ORIG}/src/code_impl/Apply_AVX.cpp"
    "${ORIG}/src/code_impl/Kalman_AVX.cpp"
    "${ORIG}/src/code_impl/Sharpen_AVX.cpp"
    PROPERTIES COMPILE_FLAGS "/arch:AVX")
  set_source_files_properties(
    "${ORIG}/src/code_impl/Apply_AVX512.cpp"
    "${ORIG}/src/code_impl/Kalman_AVX512.cpp"
    "${ORIG}/src/code_impl/Sharpen_AVX512.cpp"
    PROPERTIES COMPILE_FLAGS "/arch:AVX512")
  target_compile_options(neo-fft3d PRIVATE /MT /O2 /EHsc)
else()
  target_compile_options(neo-fft3d PRIVATE -O2)
endif()
