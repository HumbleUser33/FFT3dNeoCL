# Standalone CPU reference harness: original neo_FFT3D engine, no plugin API.
cmake_minimum_required(VERSION 3.15)
project(fft3d_ref CXX)

set(ORIG "${CMAKE_CURRENT_SOURCE_DIR}/../..")

set(SRC
  "fft3d_ref.cpp"
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

add_executable(fft3d_ref ${SRC})
set_property(TARGET fft3d_ref PROPERTY CXX_STANDARD 17)

# stubs first: replaces dualsynth types; dualsynth dir is NOT on the path
target_include_directories(fft3d_ref PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/stubs"
  "${ORIG}/include/avisynth"
  "${ORIG}"
  "${ORIG}/src"
)

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
  target_compile_options(fft3d_ref PRIVATE /MT /O2 /EHsc)
else()
  target_compile_options(fft3d_ref PRIVATE -O2)
endif()

# stage dumps are compiled in; active only when FFT3D_DUMPDIR env is set
target_compile_definitions(fft3d_ref PRIVATE FFT3D_DUMP)

# copy fftw single-precision dll next to the executable
add_custom_command(TARGET fft3d_ref POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/fftw/libfftw3f-3.dll"
    "$<TARGET_FILE_DIR:fft3d_ref>/libfftw3f-3.dll")
