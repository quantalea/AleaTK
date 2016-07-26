using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using NUnit.Framework;

namespace AleaTKTest
{
    public static class Common
    {
        // Specify the gpu id which will be used in tests,
        // so that we can change this value to test on different gpu.
        public const int GpuId = 0;

        // Specify the stream id which will be used in tests,
        // so that we can change this value to test on different gpu.
        public const int StreamId = 1;
    }
}
