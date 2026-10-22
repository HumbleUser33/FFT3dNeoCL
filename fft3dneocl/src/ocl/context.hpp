// OpenCL context: device choice, program build, readable errors.
#pragma once

#include <string>
#include <vector>
#include <cstdio>
#include <map>
#include <mutex>
#include "ocl_loader.hpp"

namespace ocl {

struct DeviceInfo {
  cl_platform_id platform {};
  cl_device_id device {};
  std::string platform_name, device_name, version;
  cl_ulong local_mem {};
  cl_ulong global_mem {};
  cl_uint max_wg {};
};

inline std::string err_str(cl_int e) {
  char b[64]; snprintf(b, sizeof(b), "CL error %d", (int)e); return b;
}

// One OpenCL context per device, shared by every filter instance (fft3dcl does
// the same with its global GpuContext). Creating and destroying a fresh context
// per instance makes the driver accumulate host (DRAM) staging memory that is
// never returned: the StaxRip preview open/close leak. The pool keeps contexts
// alive for the process lifetime; buffers/programs/queues stay per-instance.
inline cl_context shared_context(Api& api, cl_device_id dev, char* errbuf, size_t errlen) {
  struct Pool {
    std::mutex mtx;
    std::map<cl_device_id, cl_context> m;
    ~Pool() {
      auto& a = ocl::shared_api(); // constructed before the pool: still alive here
      for (auto& kv : m) if (kv.second && a.ReleaseContext) a.ReleaseContext(kv.second);
    }
  };
  static Pool pool;
  std::lock_guard<std::mutex> lk(pool.mtx);
  auto it = pool.m.find(dev);
  if (it != pool.m.end()) return it->second;
  cl_int e;
  cl_context c = api.CreateContext(nullptr, 1, &dev, nullptr, nullptr, &e);
  if (e != CL_SUCCESS) { snprintf(errbuf, errlen, "CreateContext: %s", err_str(e).c_str()); return nullptr; }
  pool.m.emplace(dev, c);
  return c;
}

// One command queue per (context, device): the driver's host-side staging
// arena follows the queue; recreating it per filter instance re-ratchets it.
inline cl_command_queue shared_queue(Api& api, cl_context ctx, cl_device_id dev, char* errbuf, size_t errlen) {
  struct QPool {
    std::mutex mtx;
    std::map<std::pair<cl_context, cl_device_id>, cl_command_queue> m;
    ~QPool() {
      auto& a = ocl::shared_api();
      for (auto& kv : m) if (kv.second && a.ReleaseCommandQueue) a.ReleaseCommandQueue(kv.second);
    }
  };
  static QPool qp;
  std::lock_guard<std::mutex> lk(qp.mtx);
  auto key = std::make_pair(ctx, dev);
  auto it = qp.m.find(key);
  if (it != qp.m.end()) return it->second;
  cl_int e;
  cl_command_queue q = api.CreateCommandQueue(ctx, dev, 0, &e);
  if (e != CL_SUCCESS) { snprintf(errbuf, errlen, "CreateCommandQueue: %s", err_str(e).c_str()); return nullptr; }
  qp.m.emplace(key, q);
  return q;
}

// device enumeration is stable within a process; enumerate once
inline const std::vector<DeviceInfo>& cached_devices(Api& api);

class Context {
public:
  Api& api = shared_api();
  cl_context ctx {nullptr};
  cl_command_queue queue {nullptr};
  cl_program program {nullptr};
  DeviceInfo info;

  // list all GPU (and optionally CPU) devices
  static std::vector<DeviceInfo> list_devices(Api& api, bool include_cpu = false) {
    std::vector<DeviceInfo> out;
    cl_uint npf = 0;
    if (api.GetPlatformIDs(0, nullptr, &npf) != CL_SUCCESS || npf == 0) return out;
    std::vector<cl_platform_id> pf(npf);
    api.GetPlatformIDs(npf, pf.data(), nullptr);
    for (cl_platform_id p : pf) {
      cl_uint nd = 0;
      cl_device_type type = CL_DEVICE_TYPE_GPU | (include_cpu ? CL_DEVICE_TYPE_CPU : 0);
      if (api.GetDeviceIDs(p, type, 0, nullptr, &nd) != CL_SUCCESS) continue;
      std::vector<cl_device_id> dv(nd);
      api.GetDeviceIDs(p, type, nd, dv.data(), nullptr);
      for (cl_device_id d : dv) {
        DeviceInfo di;
        di.platform = p; di.device = d;
        char buf[512];
        api.GetPlatformInfo(p, CL_PLATFORM_NAME, sizeof(buf), buf, nullptr); di.platform_name = buf;
        api.GetDeviceInfo(d, CL_DEVICE_NAME, sizeof(buf), buf, nullptr); di.device_name = buf;
        api.GetDeviceInfo(d, CL_DEVICE_VERSION, sizeof(buf), buf, nullptr); di.version = buf;
        api.GetDeviceInfo(d, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(di.local_mem), &di.local_mem, nullptr);
        api.GetDeviceInfo(d, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(di.global_mem), &di.global_mem, nullptr);
        api.GetDeviceInfo(d, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(di.max_wg), &di.max_wg, nullptr);
        out.push_back(di);
      }
    }
    return out;
  }

  // pick gpu_index-th GPU device; platform_id -1 = first platform with GPUs
  bool create(const std::string& source, const char* build_opts,
              int gpu_index, int platform_id, char* errbuf, size_t errlen) {
    if (!api.load(errbuf, errlen)) return false;

    std::vector<DeviceInfo> devs = cached_devices(api);
    if (devs.empty()) { snprintf(errbuf, errlen, "no OpenCL GPU device found"); return false; }

    if (platform_id >= 0) {
      // match by platform index in the platform list
      cl_uint npf = 0; api.GetPlatformIDs(0, nullptr, &npf);
      std::vector<cl_platform_id> pf(npf); api.GetPlatformIDs(npf, pf.data(), nullptr);
      if (platform_id >= (int)npf) { snprintf(errbuf, errlen, "platform_id out of range"); return false; }
      std::vector<DeviceInfo> f;
      for (auto& d : devs) if (d.platform == pf[platform_id]) f.push_back(d);
      devs.swap(f);
      if (devs.empty()) { snprintf(errbuf, errlen, "no GPU on platform %d", platform_id); return false; }
    }
    if (gpu_index >= (int)devs.size()) { snprintf(errbuf, errlen, "device_id out of range"); return false; }
    info = devs[gpu_index];

    cl_int e;
    ctx = shared_context(api, info.device, errbuf, errlen);
    if (!ctx) return false;
    queue = shared_queue(api, ctx, info.device, errbuf, errlen);
    if (!queue) return false;

    const char* src = source.c_str();
    size_t len = source.size();
    program = api.CreateProgramWithSource(ctx, 1, &src, &len, &e);
    if (e != CL_SUCCESS) { snprintf(errbuf, errlen, "CreateProgramWithSource: %s", err_str(e).c_str()); return false; }
    e = api.BuildProgram(program, 1, &info.device, build_opts, nullptr, nullptr);
    if (e != CL_SUCCESS) {
      size_t n = 0;
      api.GetProgramBuildInfo(program, info.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
      std::vector<char> log(n + 1);
      api.GetProgramBuildInfo(program, info.device, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
      snprintf(errbuf, errlen, "BuildProgram failed:\n%s", log.data());
      return false;
    }
    return true;
  }

  cl_kernel kernel(const char* name, cl_int* err) {
    return api.CreateKernel(program, name, err);
  }

  ~Context() {
    if (!api.lib) return;
    if (program) api.ReleaseProgram(program);
    // queue and ctx are owned by the shared pools, not released here
  }
};

// device enumeration is stable within a process; enumerate once
inline const std::vector<DeviceInfo>& cached_devices(Api& api) {
  static std::vector<DeviceInfo> devs;
  static std::once_flag once;
  std::call_once(once, [&] { devs = Context::list_devices(api, false); });
  return devs;
}

} // namespace ocl
