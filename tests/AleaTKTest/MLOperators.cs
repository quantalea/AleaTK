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
        public static void Gradient_Dot_GPU()
        {
            var rng = new Random();
            var m = 10;
            var k = 5;
            var n = 3;
            var x = Variable<double>();
            var y = Variable<double>();
            var z = Dot(x, y);

            var ctx = gpu;
            var exe = new Executor(ctx, z) { AssignAllGradient = true };

            var l = 10;
            var hx = new double[m, k];
            var hy = new double[k, n];
            UniformRandomArray(hx, rng);
            UniformRandomArray(hy, rng);
            var hz = Dot(hx, hy);
            //for (var i = 0; i < l; ++i) hz[i] = hx[i] + hy[i];
            //hx.AsTensor().Print();
            //hy.AsTensor().Print();

            exe.AssignTensor(x, hx.AsTensor());
            exe.AssignTensor(y, hy.AsTensor());
            exe.Forward();
            var tz = exe.GetTensor(z);
            //tz.Print();
            AreClose(hz, tz.ToArray2D(), 1e-10);

            var hdz = new double[m, n];
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

            AreClose(tdx.ToArray(), hdx.ToArray(), 1e-6);
            AreClose(tdy.ToArray(), hdy.ToArray(), 1e-6);
        }

        [Test]
        public static void Gradient_Add_VectorVector_GPU()
        {
            var rng = new Random();
            var x = Variable<double>();
            var y = Variable<double>();
            var z = x + y;

            var ctx = gpu;
            var exe = new Executor(ctx, z) {AssignAllGradient = true};

            var l = 10;
            var hx = new double[l];
            var hy = new double[l];
            var hz = new double[l];
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

            var hdz = new double[l];
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

            AreClose(tdx.ToArray(), hdx.ToArray(), 1e-6);
        }

        [Test, Ignore("Bug")]
        public static void Gradient_Add_ScalarVector_GPU()
        {
            var rng = new Random();
            var x = Variable<double>();
            var y = Variable<double>();
            var z = x + y;

            var ctx = gpu;
            var exe = new Executor(ctx, z) { AssignAllGradient = true };

            var l = 10;
            var hx = rng.NextDouble();
            var hy = new double[l];
            var hz = new double[l];
            UniformRandomArray(hy, rng);
            for (var i = 0; i < l; ++i) hz[i] = hx + hy[i];
            //hx.AsTensor().Print();
            //hy.AsTensor().Print();

            exe.AssignTensor(x, (new[] { hx }).AsTensor());
            exe.AssignTensor(y, hy.AsTensor());
            exe.Forward();
            var tz = exe.GetTensor(z);
            //tz.Print();
            AreClose(hz, tz.ToArray(), 1e-10);

            var hdz = new double[l];
            UniformRandomArray(hdz, rng);
            //hdz.AsTensor().Print();
            exe.AssignGradient(z, hdz.AsTensor());
            exe.Backward();
            var tdx = exe.GetGradient(x);
            var tdy = exe.GetGradient(y);
            tdx.Print();
            tdy.Print();

            //var bump = 1e-6;
            //var hdx = GradientChecker.FiniteDifferenceGradient(exe, x, bump: bump);
            //var hdy = GradientChecker.FiniteDifferenceGradient(exe, y, bump: bump);
            //hdx.Print();
            //hdy.Print();

            //AreClose(tdx.ToArray(), hdx.ToArray(), 1e-6);
        }

        [Test, Ignore("Bug")]
        public static void Gradient_Add_VectorMatrix_GPU()
        {
            var rng = new Random();
            var x = Variable<double>();
            var y = Variable<double>();
            var z = x + y;

            var ctx = gpu;
            var exe = new Executor(ctx, z) { AssignAllGradient = true };

            var l = 10;
            var hx = rng.NextDouble();
            var hy = new double[l];
            var hz = new double[l];
            UniformRandomArray(hy, rng);
            for (var i = 0; i < l; ++i) hz[i] = hx + hy[i];
            //hx.AsTensor().Print();
            //hy.AsTensor().Print();

            exe.AssignTensor(x, (new[] { hx }).AsTensor());
            exe.AssignTensor(y, hy.AsTensor());
            exe.Forward();
            var tz = exe.GetTensor(z);
            //tz.Print();
            AreClose(hz, tz.ToArray(), 1e-10);

            var hdz = new double[l];
            UniformRandomArray(hdz, rng);
            //hdz.AsTensor().Print();
            exe.AssignGradient(z, hdz.AsTensor());
            exe.Backward();
            var tdx = exe.GetGradient(x);
            var tdy = exe.GetGradient(y);
            tdx.Print();
            tdy.Print();

            //var bump = 1e-6;
            //var hdx = GradientChecker.FiniteDifferenceGradient(exe, x, bump: bump);
            //var hdy = GradientChecker.FiniteDifferenceGradient(exe, y, bump: bump);
            //hdx.Print();
            //hdy.Print();

            //AreClose(tdx.ToArray(), hdx.ToArray(), 1e-6);
        }
    }
}
