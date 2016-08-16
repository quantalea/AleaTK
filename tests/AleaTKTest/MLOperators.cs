using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AleaTK;
using AleaTK.ML;
using NUnit.Framework;
using Context = AleaTK.Context;
using static AleaTK.Library;
using static AleaTK.ML.Library;
using static AleaTKUtil.Common;
using static AleaTKTest.Common;

namespace AleaTKTest
{
    public static class MLOperators
    {
        private static readonly Context cpu = Context.CpuContext;
        private static readonly Context gpu = Context.GpuContext(GpuId, StreamId);

        [Test]
        public static void GradientAdd1D_SameShape_GPU()
        {
            var rng = new Random();
            var x = Variable<float>();
            var y = Variable<float>();
            var z = x + y;

            var ctx = gpu;
            var exe = new Executor(ctx, z) {AssignAllGradient = true};

            var l = 10;
            var hx = new float[l];
            var hy = new float[l];
            var hz = new float[l];
            UniformRandomArray(hx, rng);
            UniformRandomArray(hy, rng);
            for (var i = 0; i < l; ++i) hz[i] = hx[i] + hy[i];
            //hx.AsTensor().Print();
            //hy.AsTensor().Print();

            exe.AssignTensor(x, hx.AsTensor());
            exe.AssignTensor(y, hy.AsTensor());
            exe.Forward();
            var tz = exe.GetTensor(z);
            //tz.Print();
            AreClose(hz, tz.ToArray(), 1e-10);

            var hdz = new float[l];
            UniformRandomArray(hdz, rng);
            //hdz.AsTensor().Print();
            exe.AssignGradient(z, hdz.AsTensor());
            exe.Backward();
            var tdx = exe.GetGradient(x);
            var tdy = exe.GetGradient(y);
            tdx.Print();
            tdy.Print();

            var bump = 1e-6;
            var hdx = GradientChecker.FiniteDifferenceGradient(exe, x, bump: bump);
            var hdy = GradientChecker.FiniteDifferenceGradient(exe, y, bump: bump);
            hdx.Print();
            hdy.Print();

            AreClose(tdx.ToArray(), hdx.ToArray(), 1e-1);

        }
    }
}
