// Stub of dualsynth DSFormat for the standalone reference harness.
// Same fields and defaults as the real one, without VS/AVS constructors.
#pragma once

struct DSFormat
{
  bool IsFamilyYUV {true}, IsFamilyRGB {false}, IsFamilyYCC {false};
  bool IsInteger {true}, IsFloat {false};
  int SSW {0}, SSH {0};
  int BitsPerSample {8}, BytesPerSample {1};
  int Planes {3};
  DSFormat() {}
};
