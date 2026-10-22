// Minimal OpenCL 1.2 loader: loads the ICD loader at runtime (OpenCL.dll /
// libOpenCL.so.1). No vendor SDK needed. CL_TARGET_OPENCL_VERSION 120.
#pragma once

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ocl {

struct Api {
  CL_API_ENTRY cl_int(CL_API_CALL* GetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
  CL_API_ENTRY cl_int(CL_API_CALL* GetPlatformInfo)(cl_platform_id, cl_platform_info, size_t, void*, size_t*);
  CL_API_ENTRY cl_int(CL_API_CALL* GetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
  CL_API_ENTRY cl_int(CL_API_CALL* GetDeviceInfo)(cl_device_id, cl_device_info, size_t, void*, size_t*);
  CL_API_ENTRY cl_context(CL_API_CALL* CreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*, void(CL_API_CALL*)(const char*, size_t, void*, void*), void*, cl_int*);
  CL_API_ENTRY cl_int(CL_API_CALL* ReleaseContext)(cl_context);
  CL_API_ENTRY cl_command_queue(CL_API_CALL* CreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
  CL_API_ENTRY cl_int(CL_API_CALL* ReleaseCommandQueue)(cl_command_queue);
  CL_API_ENTRY cl_program(CL_API_CALL* CreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
  CL_API_ENTRY cl_int(CL_API_CALL* BuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void(CL_API_CALL*)(cl_program, void*), void*);
  CL_API_ENTRY cl_int(CL_API_CALL* GetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*);
  CL_API_ENTRY cl_int(CL_API_CALL* ReleaseProgram)(cl_program);
  CL_API_ENTRY cl_kernel(CL_API_CALL* CreateKernel)(cl_program, const char*, cl_int*);
  CL_API_ENTRY cl_int(CL_API_CALL* ReleaseKernel)(cl_kernel);
  CL_API_ENTRY cl_int(CL_API_CALL* SetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
  CL_API_ENTRY cl_mem(CL_API_CALL* CreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
  CL_API_ENTRY cl_int(CL_API_CALL* ReleaseMemObject)(cl_mem);
  CL_API_ENTRY cl_int(CL_API_CALL* EnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
  CL_API_ENTRY cl_int(CL_API_CALL* EnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*);
  CL_API_ENTRY cl_int(CL_API_CALL* EnqueueCopyBuffer)(cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, const cl_event*, cl_event*);
  CL_API_ENTRY cl_int(CL_API_CALL* EnqueueFillBuffer)(cl_command_queue, cl_mem, const void*, size_t, size_t, size_t, cl_uint, const cl_event*, cl_event*);
  CL_API_ENTRY void*(CL_API_CALL* EnqueueMapBuffer)(cl_command_queue, cl_mem, cl_bool, cl_map_flags, size_t, size_t, cl_uint, const cl_event*, cl_event*, cl_int*);
  CL_API_ENTRY cl_int(CL_API_CALL* EnqueueUnmapMemObject)(cl_command_queue, cl_mem, void*, cl_uint, const cl_event*, cl_event*);
  CL_API_ENTRY cl_int(CL_API_CALL* EnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
  CL_API_ENTRY cl_int(CL_API_CALL* Finish)(cl_command_queue);
  CL_API_ENTRY cl_int(CL_API_CALL* Flush)(cl_command_queue);

  void* lib {nullptr};

  bool load(char* errbuf, size_t errlen) {
    if (lib) return true; // already loaded (shared_api singleton)
#ifdef _WIN32
    lib = (void*)LoadLibraryA("OpenCL.dll");
    const char* missing = "OpenCL.dll not found (install GPU driver)";
#else
    lib = dlopen("libOpenCL.so.1", RTLD_NOW);
    const char* missing = "libOpenCL.so.1 not found (install ocl-icd)";
#endif
    if (!lib) { snprintf(errbuf, errlen, "%s", missing); return false; }

    auto get = [&](const char* n) -> void* {
#ifdef _WIN32
      return (void*)GetProcAddress((HMODULE)lib, n);
#else
      return dlsym(lib, n);
#endif
    };
#define OCL_GET(name) \
    do { \
      void* p = get("cl" #name); \
      if (!p) { snprintf(errbuf, errlen, "missing function: cl" #name); return false; } \
      name = reinterpret_cast<decltype(name)>(p); \
    } while (0)
    OCL_GET(GetPlatformIDs);
    OCL_GET(GetPlatformInfo);
    OCL_GET(GetDeviceIDs);
    OCL_GET(GetDeviceInfo);
    OCL_GET(CreateContext);
    OCL_GET(ReleaseContext);
    OCL_GET(CreateCommandQueue);
    OCL_GET(ReleaseCommandQueue);
    OCL_GET(CreateProgramWithSource);
    OCL_GET(BuildProgram);
    OCL_GET(GetProgramBuildInfo);
    OCL_GET(ReleaseProgram);
    OCL_GET(CreateKernel);
    OCL_GET(ReleaseKernel);
    OCL_GET(SetKernelArg);
    OCL_GET(CreateBuffer);
    OCL_GET(ReleaseMemObject);
    OCL_GET(EnqueueReadBuffer);
    OCL_GET(EnqueueWriteBuffer);
    OCL_GET(EnqueueCopyBuffer);
    OCL_GET(EnqueueFillBuffer);
    OCL_GET(EnqueueMapBuffer);
    OCL_GET(EnqueueUnmapMemObject);
    OCL_GET(EnqueueNDRangeKernel);
    OCL_GET(Finish);
    OCL_GET(Flush);
#undef _WIN_OR_POSIX_GET
#undef OCL_GET
    return true;
  }
#undef OCL_GET
};

// One OpenCL loader per process: OpenCL.dll is loaded once and kept mapped.
// Per-instance LoadLibrary churn is unnecessary and the driver's host-side
// arenas persist for the process lifetime anyway.
inline Api& shared_api() {
  static Api a;
  return a;
}

} // namespace ocl
