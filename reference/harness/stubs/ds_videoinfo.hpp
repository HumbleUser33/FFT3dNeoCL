// Stub of dualsynth DSVideoInfo for the standalone reference harness.
#pragma once

struct DSVideoInfo
{
  DSFormat Format;
  int64_t FPSNum {1}, FPSDenom {1};
  int Width {0}, Height {0};
  int Frames {0};

  int Audio_SPS {0};
  int Audio_SType {0};
  int64_t Audio_NSamples {0};
  int Audio_NChannels {0};

  int Field {0};

  DSVideoInfo() {}
  DSVideoInfo(DSFormat format, int64_t fpsnum, int64_t fpsdenom, int width, int height, int frames)
    : Format(format)
    , FPSNum(fpsnum), FPSDenom(fpsdenom)
    , Width(width), Height(height)
    , Frames(frames)
  { }
};
