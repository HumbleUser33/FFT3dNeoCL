// Stub of dualsynth ds_common.hpp for the standalone reference harness.
// Keeps the pieces the engine uses: GlobalLockGuard (mutex only, no AVS env)
// and FetchFrameFunctor.
#pragma once

#include "avisynth.h"
#include <cstring>
#include <cmath>
#include <string>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <mutex>
#include "ds_format.hpp"
#include "ds_videoinfo.hpp"
#include "ds_frame.hpp"

inline std::mutex& GetFFTWMutex() {
    static std::mutex m;
    return m;
}

class GlobalLockGuard
{
    bool acquired;
public:
    GlobalLockGuard(IScriptEnvironment* _env, const char* _name, bool use_avs_lock)
        : acquired(false)
    {
        if (!name_ok(_name))
            return;
        GetFFTWMutex().lock();
        acquired = true;
    }
    ~GlobalLockGuard()
    {
        if (acquired)
            GetFFTWMutex().unlock();
    }
    GlobalLockGuard(const GlobalLockGuard&) = delete;
    GlobalLockGuard& operator=(const GlobalLockGuard&) = delete;

private:
    static bool name_ok(const char* n) { return n != nullptr; }
};

struct FetchFrameFunctor
{
  virtual DSFrame operator()(int n) = 0;
  virtual ~FetchFrameFunctor() {}
};
