using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using AleaTK;
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

        public static void AreClose(Tensor<float> expected, Tensor<float> actual, double error)
        {
            var equalShape = expected.Shape.SequenceEqual(actual.Shape);
            if (!equalShape) Assert.Fail($"Shapes don't match");
            var expectedArray = expected.Reshape(-1).ToArray();
            var actualArray = actual.Reshape(-1).ToArray();
            AleaTKUtil.Common.AreClose(expectedArray, actualArray, error);
        }
    }
}
