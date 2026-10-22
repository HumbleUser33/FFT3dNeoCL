cmake_minimum_required(VERSION 3.15)
project(fft3dneocl CXX)

set(CMAKE_CXX_STANDARD 17)

# Khronos headers (fetched in third_party) and VapourSynth API4 headers
set(HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/../third_party")

# --- GPU FFT unit test (stage dumps from the CPU reference harness) ---
add_executable(fft_test tools/fft_test.cpp)
target_include_directories(fft_test PRIVATE "${HEADERS}/OpenCL-Headers")
if(WIN32)
  target_compile_options(fft_test PRIVATE /MT /O2 /EHsc)
else()
  target_link_libraries(fft_test dl)
endif()

# --- VapourSynth API4 plugin (headers only, no import library needed) ---
# The .cl kernels are embedded into the binary at build time.
set(KERNEL_GEN "${CMAKE_BINARY_DIR}/gen/kernel_source_gen.hpp")
add_custom_command(
  OUTPUT "${KERNEL_GEN}"
  COMMAND ${CMAKE_COMMAND}
    -DCL_SRC=${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/fft3d.cl
    -DOUT=${KERNEL_GEN}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_kernel.cmake
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/fft3d.cl
          ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_kernel.cmake)
add_library(fft3dneocl SHARED src/plugin.cpp src/gpu/pipeline.cpp "${KERNEL_GEN}")
target_include_directories(fft3dneocl PRIVATE
  "${HEADERS}/OpenCL-Headers" "${HEADERS}/vapoursynth-headers"
  "${CMAKE_BINARY_DIR}/gen")
if(WIN32)
  target_compile_options(fft3dneocl PRIVATE /MT /O2 /EHsc)
else()
  target_link_libraries(fft3dneocl dl)
endif()

# --- smoke test: drives the plugin through the VapourSynth core (loaded at runtime) ---
add_executable(plugin_smoke tools/plugin_smoke.cpp)
target_include_directories(plugin_smoke PRIVATE "${HEADERS}/vapoursynth-headers")
target_compile_definitions(plugin_smoke PRIVATE VS_USE_API_43) # clearCoreCaches (API 4.1)
if(WIN32)
  target_compile_options(plugin_smoke PRIVATE /MT /O2 /EHsc)
endif()

# --- leak harness: core create/destroy cycles, private memory per cycle ---
add_executable(leak_harness tools/leak_harness.cpp)
target_include_directories(leak_harness PRIVATE "${HEADERS}/vapoursynth-headers")
if(WIN32)
  target_compile_options(leak_harness PRIVATE /MT /O2 /EHsc)
  target_link_libraries(leak_harness psapi)
endif()

# --- host tables/geometry validator vs CPU dumps ---
add_executable(tables_check tools/tables_check.cpp)
if(WIN32)
  target_compile_options(tables_check PRIVATE /MT /O2 /EHsc)
endif()
