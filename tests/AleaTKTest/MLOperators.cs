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

        public class WeightedSumReduce<T> : Differentiable
        {
            public WeightedSumReduce(Variable<T> weights, Variable<T> vectors)
            {
                Weights = weights;
                Vectors = vectors;

                // output shape is the broadcast of w*v, skip the first dimension
                // currently the reduce is fixed on first dimension
                //if (weights.HasShape && vectors.HasShape)
                //{
                //    var shape = PartialShape.Broadcast(weights.Shape, vectors.Shape);
                //    Output = Variable<T>(PartialShape.Create(shape.Skip(1).ToArray()));
                //}
                //else
                //{
                //    Output = Variable<T>();
                //}
                Output = Variable<T>();

                AddInput(Weights);
                AddInput(Vectors);
                AddOutput(Output);
            }

            public Variable<T> Weights { get; }

            public Variable<T> Vectors { get; }

            public Variable<T> Output { get; }

            public override void Forward(Executor executor)
            {
                var vectors = executor.GetTensor(Vectors);
                var weights = executor.GetTensor(Weights);

                var prod = weights*vectors;

                // currently reduce sum only works up to 2d tensor
                // then we do a reduce to make it an 2d tensor
                // after reduce, we reshape it back.
                var length0 = prod.Shape[0];
                var shape = prod.Shape.Skip(1).ToArray();
                var reduce = ReduceSum(prod.Reshape(length0, -1), 0).Reshape(shape);

                executor.AssignTensor(Output, reduce);
            }

            public override void Backward(Executor executor)
            {
                var vectors = executor.GetTensor(Vectors);
                var weights = executor.GetTensor(Weights);
                var dOutput = executor.GetGradient(Output);

                Console.WriteLine((vectors * dOutput).Shape);
                executor.AssignGradient(Vectors, weights*dOutput);
                

                throw new Exception("TODO");
                //executor.AssignGradient(softmax, vectors*dOutput);
            }
        }

        [Test]
        public static void Gradient_WeightedSumReduce_01_GPU()
        {
            var rng = new Random(42);
            var x = Variable<double>();
            var w = Variable<double>();
            var wsr = new WeightedSumReduce<double>(w, x);
            var y = wsr.Output;

            var ctx = gpu;
            var exe = new Executor(ctx, y) {AssignAllGradient = true};

            var n = 5;
            var d = 3;
            var hx = new double[n, d];
            var hw = new double[n, d];
            UniformRandomArray(hx, rng);
            UniformRandomArray(hw, rng);
            var hy = new double[d];
            for (var i = 0; i < d; ++i)
            {
                var acc = 0.0;
                for (var j = 0; j < n; ++j)
                {
                    acc += hw[j, i]*hx[j, i];
                }
                hy[i] = acc;
            }

            exe.AssignTensor(x, hx.AsTensor());
            exe.AssignTensor(w, hw.AsTensor());
            exe.Forward();
            var ty = exe.GetTensor(y);
            ty.Print();
            AreClose(hy, ty.ToArray(), 1e-10);

            var hdy = new double[d];
            UniformRandomArray(hdy, rng);
            exe.AssignGradientDirectly(y, hdy.AsTensor());
            exe.Backward();
            var tdx = exe.GetGradient(x);
            var tdw = exe.GetGradient(w);
            tdx.Print();
            tdw.Print();

            var bump = 1e-8;
            var hdx = GradientChecker.FiniteDifferenceGradient(exe, x, bump: bump);
            var hdw = GradientChecker.FiniteDifferenceGradient(exe, w, bump: bump);
            hdx.Print();
            hdw.Print();

            AreClose(hdx.ToArray2D(), tdx.ToArray2D(), 1e-7);
            AreClose(hdw.ToArray2D(), tdw.ToArray2D(), 1e-7);
        }
        
        [Test]
        public static void Gradient_WeightedSumReduce_02_GPU()
        {
            var rng = new Random(42);
            var x = Variable<double>();
            var w = Variable<double>();
            var wsr = new WeightedSumReduce<double>(w.Reshape(-1, 1), x);
            var y = wsr.Output;

            var ctx = gpu;
            var exe = new Executor(ctx, y) { AssignAllGradient = true };

            var n = 5;
            var d = 3;
            var hx = new double[n, d];
            var hw = new double[n];
            UniformRandomArray(hx, rng);
            UniformRandomArray(hw, rng);
            var hy = new double[d];
            for (var i = 0; i < d; ++i)
            {
                var acc = 0.0;
                for (var j = 0; j < n; ++j)
                {
                    acc += hw[j] * hx[j, i];
                }
                hy[i] = acc;
            }

            exe.AssignTensor(x, hx.AsTensor());
            exe.AssignTensor(w, hw.AsTensor());
            exe.Forward();
            var ty = exe.GetTensor(y);
            ty.Print();
            AreClose(hy, ty.ToArray(), 1e-10);

            var hdy = new double[d];
            UniformRandomArray(hdy, rng);
            exe.AssignGradientDirectly(y, hdy.AsTensor());
            exe.Backward();
            //var tdx = exe.GetGradient(x);
            //var tdw = exe.GetGradient(w);
            //tdx.Print();
            //tdw.Print();

            //var bump = 1e-8;
            //var hdx = GradientChecker.FiniteDifferenceGradient(exe, x, bump: bump);
            ////var hdw = GradientChecker.FiniteDifferenceGradient(exe, w, bump: bump);
            //hdx.Print();
            ////hdw.Print();

            //AreClose(hdx.ToArray2D(), tdx.ToArray2D(), 1e-7);
            ////AreClose(hdw.ToArray2D(), tdw.ToArray2D(), 1e-7);
        }
    }
}
