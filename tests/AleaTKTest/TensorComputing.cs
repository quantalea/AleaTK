using System;
using System.Linq;
using Alea;
using AleaTK;
using NUnit.Framework;
using Context = AleaTK.Context;
using static AleaTK.Library;
using static AleaTKUtil.Common;
using static AleaTKTest.Common;

namespace AleaTKTest
{
    public static class TensorComputing
    {
        private static readonly Context cpu = Context.CpuContext;
        private static readonly Context gpu = Context.GpuContext(GpuId, StreamId);

        private static void Main()
        {
            //PiEstimationGpu();
        }

        [Test]
        public static void AssignOnes1DCpu()
        {
            var ctx = cpu;
            const int n = 1000;
            var a = ctx.Allocate(Shape.Create(n), 1.0);
            var b = ctx.Device.Allocate<double>(Shape.Create(n));
            Assert.IsTrue(a.ToArray().SequenceEqual(Enumerable.Repeat(1.0, n)));
            ctx.Assign(b, a);
            Assert.IsTrue(b.ToArray().SequenceEqual(Enumerable.Repeat(1.0, n)));
            b.Print();
        }

        [Test]
        public static void AssignOnes1DGpu()
        {
            var ctx = gpu;
            const int n = 1000;
            var a = ctx.Allocate(Shape.Create(n), 1.0);
            var b = ctx.Device.Allocate<double>(Shape.Create(n));
            Assert.IsTrue(a.ToArray().SequenceEqual(Enumerable.Repeat(1.0, n)));
            ctx.Assign(b, a);
            Assert.IsTrue(b.ToArray().SequenceEqual(Enumerable.Repeat(1.0, n)));
            b.Print();
        }

        [Test]
        public static void AssignOnes2DCpu()
        {
            var ctx = cpu;
            const int m = 100;
            const int n = 50;
            var a = ctx.Device.Allocate<double>(Shape.Create(m, n));
            ctx.Assign(a, 3.14);
            a.Print();
            var expected = new double[m, n];
            for (var i = 0; i < m; ++i)
                for (var j = 0; j < n; ++j)
                    expected[i, j] = 3.14;
            var actual = a.ToArray2D();
            AreEqual(expected, actual);
        }

        [Test]
        public static void AssignOnes2DGpu()
        {
            var ctx = gpu;
            const int m = 100;
            const int n = 50;
            var a = ctx.Device.Allocate<double>(Shape.Create(m, n));
            ctx.Assign(a, 3.14);
            a.Print();
            var expected = new double[m, n];
            for (var i = 0; i < m; ++i)
                for (var j = 0; j < n; ++j)
                    expected[i, j] = 3.14;
            var actual = a.ToArray2D();
            AreEqual(expected, actual);
        }

        [Test]
        public static void ReferenceThenAssignCpu()
        {
            var ctx = cpu;
            const int n = 1000;
            var arrayA = GenerateRandomDoubleData(n, 1, 100);
            var arrayB = new double[n];
            var a = arrayA.AsTensor();
            var b = arrayB.AsTensor();
            // since b is just reference of a host array, so we need run
            // assignment sync (last argument sync = true)
            ctx.Assign(b, a).Wait();
            b.Print();
            Assert.IsTrue(arrayB.SequenceEqual(arrayA));
        }

        [Test]
        public static void ReferenceThenCopyThenAssignGpu()
        {
            const int n = 1000;
            var arrayA = GenerateRandomDoubleData(n, 1, 100);
            var arrayB = new double[n];
            var cpuA = arrayA.AsTensor();
            var cpuB = arrayB.AsTensor();
            var gpuA = gpu.Device.Allocate<double>(Shape.Create(n));
            var gpuB = gpu.Device.Allocate<double>(Shape.Create(n));
            gpu.Copy(gpuA, cpuA);
            gpu.Assign(gpuB, gpuA);
            // this copy need to sync, since cpuB is just a reference
            gpu.Copy(cpuB, gpuB).Wait();
            gpuB.Print();
            Assert.IsTrue(arrayB.SequenceEqual(arrayA));
        }

        [Test]
        public static void AllocateTensorWithInitValuesCpu()
        {
            var ctx = cpu;
            const int n = 1000;
            var array = GenerateRandomDoubleData(n, 1, 100);
            var a = ctx.Allocate(array);
            var b = ctx.Device.Allocate<double>(Shape.Create(n));
            ctx.Assign(b, a);
            b.Print();
            Assert.IsTrue(b.ToArray().SequenceEqual(array));
        }

        [Test]
        public static void AllocateTensorWithInitValuesGpu()
        {
            var ctx = gpu;
            const int n = 1000;
            var array = GenerateRandomDoubleData(n, 1, 100);
            var a = ctx.Allocate(array);
            var b = ctx.Device.Allocate<double>(Shape.Create(n));
            ctx.Assign(b, a);
            b.Print();
            Assert.IsTrue(b.ToArray().SequenceEqual(array));
        }

        [Test]
        public static void SimpleMathCpu()
        {
            var ctx = cpu;
            const int n = 1000;
            var input = GenerateRandomDoubleData(n, -2, 2);
            var a = ctx.Allocate(input);
            //a.Print();
            ctx.Assign(a, 3.0 * a + Exp(a + 1.5));
            a.Print();
            var expected = input.Select(x => 3.0 * x + Math.Exp(x + 1.5)).ToArray();
            var actual = a.ToArray();
            Assert.AreEqual(expected, actual);
        }

        [Test]
        public static void SimpleMathGpu()
        {
            var ctx = gpu;
            const int n = 1000;
            var input = GenerateRandomDoubleData(n, -2, 2);
            var a = ctx.Allocate(input);
            //a.Print();
            ctx.Assign(a, 3.0 * a + Exp(a + 1.5));
            a.Print();
            var expected = input.Select(x => 3.0 * x + Math.Exp(x + 1.5)).ToArray();
            var actual = a.ToArray();
            AreClose(expected, actual, 1e-10);
        }

        [Test]
        public static void SigmoidCpu()
        {
            var ctx = cpu;
            const int n = 1000;
            var input = GenerateRandomDoubleData(n, -2, 2);
            var a = ctx.Allocate(input);
            //a.Print();
            ctx.Assign(a, 1.0 / (1.0 + Exp(-a)));
            a.Print();
            var expected = input.Select(x => 1.0 / (1.0 + Math.Exp(-x))).ToArray();
            var actual = a.ToArray();
            Assert.AreEqual(expected, actual);
        }

        [Test]
        public static void SigmoidGpu()
        {
            var ctx = gpu;
            const int n = 1000;
            var input = GenerateRandomDoubleData(n, -2, 2);
            var a = ctx.Allocate(input);
            //a.Print();
            ctx.Assign(a, 1.0 / (1.0 + Exp(-a)));
            a.Print();
            var expected = input.Select(x => 1.0 / (1.0 + Math.Exp(-x))).ToArray();
            var actual = a.ToArray();
            AreClose(expected, actual, 1e-10);
        }

        [Test]
        public static void RepMatOverRowsCpu()
        {
            var ctx = cpu;
            var input = new[] { 1.0, 2.0, 3.0 };
            // a is rank 1, when we assign it to rank 2, by boradcasting
            // rule, it will be implicitly reshape to [1,3], which means,
            // repeat mat over rows.
            var a = ctx.Allocate(input);
            var b = ctx.Device.Allocate<double>(Shape.Create(5, 3));
            a.Print();
            ctx.Assign(b, a);
            b.Print();
            var expected = new[,]
            {
                {1.0, 2.0, 3.0},
                {1.0, 2.0, 3.0},
                {1.0, 2.0, 3.0},
                {1.0, 2.0, 3.0},
                {1.0, 2.0, 3.0}
            };
            var actual = b.ToArray2D();
            AreEqual(expected, actual);
        }

        [Test]
        public static void RepMatOverRowsGpu()
        {
            var ctx = gpu;
            var input = new[] { 1.0, 2.0, 3.0 };
            // a is rank 1, when we assign it to rank 2, by boradcasting
            // rule, it will be implicitly reshape to [1,3], which means,
            // repeat mat over rows.
            var a = ctx.Allocate(input);
            var b = ctx.Device.Allocate<double>(Shape.Create(5, 3));
            a.Print();
            ctx.Assign(b, a);
            b.Print();
            var expected = new[,]
            {
                {1.0, 2.0, 3.0},
                {1.0, 2.0, 3.0},
                {1.0, 2.0, 3.0},
                {1.0, 2.0, 3.0},
                {1.0, 2.0, 3.0}
            };
            var actual = b.ToArray2D();
            AreEqual(expected, actual);
        }

        [Test]
        public static void RepMatOverColsCpu()
        {
            var ctx = cpu;
            var input = new[,] { {1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
            var a = ctx.Allocate(input);
            var b = ctx.Device.Allocate<double>(Shape.Create(5, 3));
            a.Print();
            ctx.Assign(b, a);
            b.Print();
            var expected = new[,]
            {
                {1.0, 1.0, 1.0},
                {2.0, 2.0, 2.0},
                {3.0, 3.0, 3.0},
                {4.0, 4.0, 4.0},
                {5.0, 5.0, 5.0}
            };
            var actual = b.ToArray2D();
            AreEqual(expected, actual);
        }

        [Test]
        public static void RepMatOverColsGpu()
        {
            var ctx = gpu;
            var input = new[,] { { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 }, { 5.0 } };
            var a = ctx.Allocate(input);
            var b = ctx.Device.Allocate<double>(Shape.Create(5, 3));
            a.Print();
            ctx.Assign(b, a);
            b.Print();
            var expected = new[,]
            {
                {1.0, 1.0, 1.0},
                {2.0, 2.0, 2.0},
                {3.0, 3.0, 3.0},
                {4.0, 4.0, 4.0},
                {5.0, 5.0, 5.0}
            };
            var actual = b.ToArray2D();
            AreEqual(expected, actual);
        }

        [Test]
        public static void RepMatOverColsViaReshapeCpu()
        {
            var ctx = cpu;
            var input = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
            var a = ctx.Allocate(input);
            var b = ctx.Device.Allocate<double>(Shape.Create(5, 3));
            a.Print();
            ctx.Assign(b, a.Reshape(-1, 1));
            b.Print();
            var expected = new[,]
            {
                {1.0, 1.0, 1.0},
                {2.0, 2.0, 2.0},
                {3.0, 3.0, 3.0},
                {4.0, 4.0, 4.0},
                {5.0, 5.0, 5.0}
            };
            var actual = b.ToArray2D();
            AreEqual(expected, actual);
        }

        [Test]
        public static void RepMatOverColsViaReshapeGpu()
        {
            var ctx = gpu;
            var input = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
            var a = ctx.Allocate(input);
            var b = ctx.Device.Allocate<double>(Shape.Create(5, 3));
            a.Print();
            // a is rank1, it is row vector by the broadcasting rule,
            // to make it repeat over columns, you need change it to 
            // column vector. You can do this via reshape. the -1
            // in reshapr means, it can be calculated. For more detail
            // see https://www.tensorflow.org/versions/r0.8/api_docs/python/array_ops.html#reshape
            ctx.Assign(b, a.Reshape(-1, 1));
            b.Print();
            var expected = new[,]
            {
                {1.0, 1.0, 1.0},
                {2.0, 2.0, 2.0},
                {3.0, 3.0, 3.0},
                {4.0, 4.0, 4.0},
                {5.0, 5.0, 5.0}
            };
            var actual = b.ToArray2D();
            AreEqual(expected, actual);
        }

        [Test]
        public static void RepMatOverColsViaTransposeCpu()
        {
            var ctx = cpu;
            var input = new[,] { { 1.0, 2.0, 3.0, 4.0, 5.0 } };
            var a = ctx.Allocate(input);
            var b = ctx.Device.Allocate<double>(Shape.Create(5, 3));
            a.Print();
            ctx.Assign(b, a.T);
            b.Print();
            var expected = new[,]
            {
                {1.0, 1.0, 1.0},
                {2.0, 2.0, 2.0},
                {3.0, 3.0, 3.0},
                {4.0, 4.0, 4.0},
                {5.0, 5.0, 5.0}
            };
            var actual = b.ToArray2D();
            AreEqual(expected, actual);
        }

        [Test]
        public static void RepMatOverColsViaTransposeGpu()
        {
            var ctx = gpu;
            var input = new[,] { { 1.0, 2.0, 3.0, 4.0, 5.0 } };
            var a = ctx.Allocate(input);
            var b = ctx.Device.Allocate<double>(Shape.Create(5, 3));
            a.Print();
            ctx.Assign(b, a.T);
            b.Print();
            var expected = new[,]
            {
                {1.0, 1.0, 1.0},
                {2.0, 2.0, 2.0},
                {3.0, 3.0, 3.0},
                {4.0, 4.0, 4.0},
                {5.0, 5.0, 5.0}
            };
            var actual = b.ToArray2D();
            AreEqual(expected, actual);
        }

        [Test]
        public static void ReduceSumCpu()
        {
            var ctx = cpu;
            const int n = 1000;
            var input = GenerateRandomDoubleData(n, -2, 2);
            var a = ctx.Allocate(input);
            var b = ctx.Device.Allocate<double>(Shape.Scalar);
            ctx.Assign(b, ReduceSum(a));
            b.Print();
            var actual = b.ToScalar();
            Console.WriteLine(actual);
            var expected = input.Sum();
            Assert.AreEqual(expected, actual);

            // you can also calc mean by / the numItems
            ctx.Assign(b, ReduceSum(a) / n);
            b.Print();
            actual = b.ToScalar();
            expected = input.Average();
            Assert.AreEqual(expected, actual);
        }

        [Test]
        public static void ReduceSumGpu()
        {
            var ctx = gpu;
            const int n = 1000;
            var input = GenerateRandomDoubleData(n, -2, 2);
            var a = ctx.Allocate(input);
            var b = ctx.Device.Allocate<double>(Shape.Scalar);
            ctx.Assign(b, ReduceSum(a));
            b.Print();
            var actual = b.ToScalar();
            Console.WriteLine(actual);
            var expected = input.Sum();
            Assert.That(actual, Is.EqualTo(expected).Within(1e-10));

            // you can also calc mean by / the numItems
            ctx.Assign(b, ReduceSum(a) / n);
            b.Print();
            actual = b.ToScalar();
            expected = input.Average();
            Assert.That(actual, Is.EqualTo(expected).Within(1e-10));
        }

        [Test]
        public static void ReduceSumWithParamsCpu()
        {
            // https://www.tensorflow.org/versions/r0.8/api_docs/python/math_ops.html#reduce_sum
            //# 'x' is [[1, 1, 1]
            //#         [1, 1, 1]]
            //tf.reduce_sum(x) ==> 6
            //tf.reduce_sum(x, 0) ==> [2, 2, 2]
            //tf.reduce_sum(x, 1) ==> [3, 3]
            //tf.reduce_sum(x, 1, keep_dims = True) ==> [[3], [3]]
            //tf.reduce_sum(x, [0, 1]) ==> 6
            var ctx = cpu;
            var x = ctx.Allocate(new[,] { { 1.0, 1.0, 1.0 }, { 1.0, 1.0, 1.0 } });
            Assert.AreEqual(6.0, ctx.Eval(ReduceSum(x)).ToScalar());
            Assert.AreEqual(new[] { 2.0, 2.0, 2.0 }, ctx.Eval(ReduceSum(x, 0)).ToArray());
            Assert.AreEqual(new[] { 3.0, 3.0 }, ctx.Eval(ReduceSum(x, 1)).ToArray());
            Assert.AreEqual(new[,] { { 3.0 }, { 3.0 } }, ctx.Eval(ReduceSum(x, true, 1)).ToArray2D());
            Assert.AreEqual(6.0, ctx.Eval(ReduceSum(x, 0, 1)).ToScalar());
        }

        [Test]
        public static void ReduceSumWithParamsGpu()
        {
            // https://www.tensorflow.org/versions/r0.8/api_docs/python/math_ops.html#reduce_sum
            //# 'x' is [[1, 1, 1]
            //#         [1, 1, 1]]
            //tf.reduce_sum(x) ==> 6
            //tf.reduce_sum(x, 0) ==> [2, 2, 2]
            //tf.reduce_sum(x, 1) ==> [3, 3]
            //tf.reduce_sum(x, 1, keep_dims = True) ==> [[3], [3]]
            //tf.reduce_sum(x, [0, 1]) ==> 6
            var ctx = gpu;
            var x = ctx.Allocate(new[,] { { 1.0, 1.0, 1.0 }, { 1.0, 1.0, 1.0 } });
            Assert.AreEqual(6.0, ctx.Eval(ReduceSum(x)).ToScalar());
            Assert.AreEqual(new[] { 2.0, 2.0, 2.0 }, ctx.Eval(ReduceSum(x, 0)).ToArray());
            Assert.AreEqual(new[] { 3.0, 3.0 }, ctx.Eval(ReduceSum(x, 1)).ToArray());
            Assert.AreEqual(new[,] { { 3.0 }, { 3.0 } }, ctx.Eval(ReduceSum(x, true, 1)).ToArray2D());
            Assert.AreEqual(6.0, ctx.Eval(ReduceSum(x, 0, 1)).ToScalar());
        }

        [Test]
        public static void ReduceMeanCpu()
        {
            var ctx = cpu;
            const int n = 1000;
            var input = GenerateRandomDoubleData(n, -2, 2);
            var a = ctx.Allocate(input);
            var b = ctx.Device.Allocate<double>(Shape.Scalar);
            ctx.Assign(b, ReduceMean(1.0 / (1.0 + Exp(-a))));
            b.Print();
            var actual = b.ToScalar();
            Console.WriteLine(actual);
            var expected = input.Select(x => 1.0 / (1.0 + Math.Exp(-x))).Average();
            Assert.AreEqual(expected, actual);
        }

        [Test]
        public static void ReduceMeanGpu()
        {
            var ctx = gpu;
            const int n = 1000;
            var input = GenerateRandomDoubleData(n, -2, 2);
            var a = ctx.Allocate(input);
            var b = ctx.Device.Allocate<double>(Shape.Scalar);
            ctx.Assign(b, ReduceMean(1.0 / (1.0 + Exp(-a))));
            b.Print();
            var actual = b.ToScalar();
            Console.WriteLine(actual);
            var expected = input.Select(x => 1.0 / (1.0 + Math.Exp(-x))).Average();
            Assert.That(actual, Is.EqualTo(expected).Within(1e-10));
        }

        [Test]
        public static void AssignUniformRandomCpu()
        {
            var ctx = cpu;
            const int n = 1000;
            var a = ctx.Device.Allocate<double>(Shape.Create(n));
            ctx.Assign(a, RandomUniform<double>());
            a.Print();
            // you can use eval to directly evaluate an expression, internally
            // it will allocate a tensor and return to you.
            var mean = ctx.Eval(ReduceMean(a)).ToScalar();
            Console.WriteLine(mean);
            Assert.That(mean, Is.EqualTo(0.5).Within(1e-1));
        }

        [Test]
        public static void AssignUniformRandomGpu()
        {
            var ctx = gpu;
            const int n = 1000;
            var a = ctx.Device.Allocate<double>(Shape.Create(n));
            ctx.Assign(a, RandomUniform<double>());
            a.Print();
            // you can use eval to directly evaluate an expression, internally
            // it will allocate a tensor and return to you.
            var mean = ctx.Eval(ReduceMean(a)).ToScalar();
            Console.WriteLine(mean);
            Assert.That(mean, Is.EqualTo(0.5).Within(1e-1));
        }

        public static void EstimatePi(Context ctx, int batchs, ulong batchSize, double error)
        {
            const ulong seed = 0UL;

            // allocate buffer for the generated points and a scalar to hold the simulated value of pi
            var points = ctx.Device.Allocate<double2>(Shape.Create((long)batchSize));
            var pi = ctx.Device.Allocate<double>(Shape.Scalar);

            // transform that checks if point is inside unit square or not
            // the value 4.0 is because we only simulate points in positive quadrant
            var pis = Map(points, point => (point.x * point.x + point.y * point.y) < 1.0 ? 4.0 : 0.0);

            // iterate over multiple batches
            for (var i = 0; i < batchs; ++i)
            {
                Console.WriteLine($"Batch {i}");
                // generates random numbers, apply the mapping followed by a mean reduction
                var offset = batchSize * (ulong)i;
                ctx.Assign(points, RandomUniform<double2>(seed: seed, offset: offset));
                ctx.Assign(pi, i == 0 ? ReduceMean(pis) : (pi + ReduceMean(pis)) / 2.0);
            }

            Console.WriteLine($"Pi = {pi.ToScalar()}");
            Assert.That(pi.ToScalar(), Is.EqualTo(Math.PI).Within(error));
        }

        [Test]
        public static void EstimatePiCpu()
        {
            EstimatePi(cpu, 5, 100000, 1e-2);
        }

        [Test]
        public static void EstimatePiGpu()
        {
            EstimatePi(gpu, 100, 10000000, 1e-3);
        }

        [Test]
        public static void ShapeBroadcast01()
        {
            //Image  (3d array): 256 x 256 x 3
            //Scale  (1d array):             3
            //Result (3d array): 256 x 256 x 3
            var shape1 = Shape.Create(256, 256, 3);
            var shape2 = Shape.Create(3);
            var actual = Shape.Broadcast(shape1, shape2);
            var expected = Shape.Create(256, 256, 3);
            Console.WriteLine(actual);
            Assert.IsTrue(actual.SequenceEqual(expected));
        }

        [Test]
        public static void ShapeBroadcast02()
        {
            //A      (4d array):  8 x 1 x 6 x 1
            //B      (3d array):      7 x 1 x 5
            //Result (4d array):  8 x 7 x 6 x 5
            var shape1 = Shape.Create(8, 1, 6, 1);
            var shape2 = Shape.Create(7, 1, 5);
            var actual = Shape.Broadcast(shape1, shape2);
            var expected = Shape.Create(8, 7, 6, 5);
            Console.WriteLine(actual);
            Assert.IsTrue(actual.SequenceEqual(expected));
        }

        [Test]
        public static void ShapeBroadcast03()
        {
            var shape1 = Shape.Create(2, 3);
            var shape2 = Shape.Create(2, 1, 1);
            var shape3 = Shape.Create(1, 1, 3);
            var actual = Shape.Broadcast(shape1, shape2, shape3);
            var expected = Shape.Create(2, 2, 3);
            Console.WriteLine(actual);
            Assert.IsTrue(actual.SequenceEqual(expected));
        }

        [Test]
        public static void BroadcastScalarToVectorCpu()
        {
            var ctx = cpu;
            const int n = 1000;
            var a = ctx.Allocate(Shape.Scalar, 5.0);
            var b = ctx.Allocate(Shape.Create(n), 1.0);
            ctx.Assign(b, a);
            b.Print();
            var actual = b.ToArray();
            var expected = Enumerable.Repeat(5.0, n).ToArray();
            Assert.AreEqual(expected, actual);
        }

        [Test]
        public static void BroadcastScalarToVectorGpu()
        {
            var ctx = gpu;
            const int n = 1000;
            var a = ctx.Allocate(Shape.Scalar, 5.0);
            var b = ctx.Allocate(Shape.Create(n), 1.0);
            ctx.Assign(b, a);
            b.Print();
            var actual = b.ToArray();
            var expected = Enumerable.Repeat(5.0, n).ToArray();
            Assert.AreEqual(expected, actual);
        }

        [Test]
        public static void BroadcastScalarToMatrixCpu()
        {
            var ctx = cpu;
            var a = ctx.Allocate(Shape.Scalar, 5.0);
            var b = ctx.Allocate(Shape.Create(333, 555), 1.0);
            ctx.Assign(b, a);
            b.Print();
            var actual = b.ToArray2D();
            var expected = CreateArray(333, 555, (row, col) => 5.0);
            AreEqual(expected, actual);
        }

        [Test]
        public static void BroadcastScalarToMatrixGpu()
        {
            var ctx = gpu;
            var a = ctx.Allocate(Shape.Scalar, 5.0);
            var b = ctx.Allocate(Shape.Create(333, 555), 1.0);
            ctx.Assign(b, a);
            b.Print();
            var actual = b.ToArray2D();
            var expected = CreateArray(333, 555, (row, col) => 5.0);
            AreEqual(expected, actual);
        }

        [Test]
        public static void BroadcastVectorToMatrixCpu()
        {
            var ctx = cpu;
            var a = ctx.Allocate(new[] { 1.0, 2.0, 3.0, 4.0 });
            var b = ctx.Allocate(Shape.Create(333, 4), 1.0);
            ctx.Assign(b, a);
            b.Print();
            var actual = b.ToArray2D();
            var expected = CreateArray(333, 4, (row, col) => col + 1.0);
            AreEqual(expected, actual);
        }

        [Test]
        public static void BroadcastVectorToMatrixGpu()
        {
            var ctx = gpu;
            var a = ctx.Allocate(new[] { 1.0, 2.0, 3.0, 4.0 });
            var b = ctx.Allocate(Shape.Create(333, 4), 1.0);
            ctx.Assign(b, a);
            b.Print();
            var actual = b.ToArray2D();
            var expected = CreateArray(333, 4, (row, col) => col + 1.0);
            AreEqual(expected, actual);
        }

        [Test]
        public static void SimpleDotCpu()
        {
            var ctx = cpu;
            var inputA = new[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } };
            var inputB = new[,] { { 1.0, 2.0, 3.0, 4.0 }, { 5.0, 6.0, 7.0, 8.0 }, { 9.0, 10.0, 11.0, 12.0 } };
            var a = ctx.Allocate(inputA);
            var b = ctx.Allocate(inputB);
            a.Print();
            b.Print();
            var c = ctx.Device.Allocate<double>(Shape.Create(2, 4));
            ctx.Assign(c, Dot(a, b));
            c.Print();
            var expected = Dot(inputA, inputB);
            var actual = c.ToArray2D();
            AreEqual(expected, actual);
        }

        [Test]
        public static void SimpleDotGpu()
        {
            var ctx = gpu;
            var inputA = new[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } };
            var inputB = new[,] { { 1.0, 2.0, 3.0, 4.0 }, { 5.0, 6.0, 7.0, 8.0 }, { 9.0, 10.0, 11.0, 12.0 } };
            var a = ctx.Allocate(inputA);
            var b = ctx.Allocate(inputB);
            a.Print();
            b.Print();
            var c = ctx.Device.Allocate<double>(Shape.Create(2, 4));
            ctx.Assign(c, Dot(a, b));
            c.Print();
            var expected = Dot(inputA, inputB);
            var actual = c.ToArray2D();
            AreEqual(expected, actual);
        }

        [Test]
        public static void SimpleTransposeCpu()
        {
            var ctx = cpu;
            var inputA = new[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } };
            var a = ctx.Allocate(inputA);
            var at = a.T;
            at.Print();

            var ata = ctx.Device.Allocate<double>(Shape.Create(3, 2));
            ctx.Assign(ata, at);
            ata.Print();

            var expected = new[,] { { 1.0, 4.0 }, { 2.0, 5.0 }, { 3.0, 6.0 } };
            AreEqual(expected, ata.ToArray2D());

            var actual = at.ToArray2D();
            AreEqual(expected, actual);
        }

        [Test]
        public static void SimpleTransposeGpu()
        {
            var ctx = gpu;
            var inputA = new[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } };
            var a = ctx.Allocate(inputA);
            var at = a.T;
            at.Print();

            var ata = ctx.Device.Allocate<double>(Shape.Create(3, 2));
            ctx.Assign(ata, at);
            ata.Print();

            var expected = new[,] { { 1.0, 4.0 }, { 2.0, 5.0 }, { 3.0, 6.0 } };
            AreEqual(expected, ata.ToArray2D());

            var actual = at.ToArray2D();
            AreEqual(expected, actual);
        }

        [Test]
        public static void SimpleTransDotCpu()
        {
            var ctx = cpu;
            var inputA = new[,] { { 1.0, 4.0 }, { 2.0, 5.0 }, { 3.0, 6.0 } };
            var inputB = new[,] { { 1.0, 2.0, 3.0, 4.0 }, { 5.0, 6.0, 7.0, 8.0 }, { 9.0, 10.0, 11.0, 12.0 } };
            var a = ctx.Allocate(inputA);
            var b = ctx.Allocate(inputB);
            a.Print();
            b.Print();
            var c = ctx.Device.Allocate<double>(Shape.Create(2, 4));
            ctx.Assign(c, Dot(a.T, b));
            c.Print();
            var expected = Dot(Transpose(inputA), inputB);
            var actual = c.ToArray2D();
            AreEqual(expected, actual);
        }

        [Test]
        public static void SimpleTransDotGpu()
        {
            var ctx = gpu;
            var inputA = new[,] { { 1.0, 4.0 }, { 2.0, 5.0 }, { 3.0, 6.0 } };
            var inputB = new[,] { { 1.0, 2.0, 3.0, 4.0 }, { 5.0, 6.0, 7.0, 8.0 }, { 9.0, 10.0, 11.0, 12.0 } };
            var a = ctx.Allocate(inputA);
            var b = ctx.Allocate(inputB);
            a.Print();
            b.Print();
            var c = ctx.Device.Allocate<double>(Shape.Create(2, 4));
            ctx.Assign(c, Dot(a.T, b));
            c.Print();
            var expected = Dot(Transpose(inputA), inputB);
            var actual = c.ToArray2D();
            AreEqual(expected, actual);
        }

        [Test]
        public static void ReduceSumByRowCpu()
        {
            var ctx = cpu;
            var input = new[,] { { 1.0, 2.0, 3.0, 4.0 }, { 5.0, 6.0, 7.0, 8.0 }, { 9.0, 10.0, 11.0, 12.0 } };
            var a = ctx.Allocate(input);
            a.Print();
            var b = ctx.Device.Allocate<double>(Shape.Create(4));
            ctx.Assign(b, ReduceSum(a, 0));
            b.Print();
            var expected = new double[input.GetLength(1)];
            expected[0] = 1.0 + 5.0 + 9.0;
            expected[1] = 2.0 + 6.0 + 10.0;
            expected[2] = 3.0 + 7.0 + 11.0;
            expected[3] = 4.0 + 8.0 + 12.0;
            var actual = b.ToArray();
            Assert.AreEqual(expected, actual);
        }

        [Test]
        public static void ReduceSumByRowGpu()
        {
            var ctx = gpu;
            var input = new[,] { { 1.0, 2.0, 3.0, 4.0 }, { 5.0, 6.0, 7.0, 8.0 }, { 9.0, 10.0, 11.0, 12.0 } };
            var a = ctx.Allocate(input);
            a.Print();
            var b = ctx.Device.Allocate<double>(Shape.Create(4));
            ctx.Assign(b, ReduceSum(a, 0));
            b.Print();
            var expected = new double[input.GetLength(1)];
            expected[0] = 1.0 + 5.0 + 9.0;
            expected[1] = 2.0 + 6.0 + 10.0;
            expected[2] = 3.0 + 7.0 + 11.0;
            expected[3] = 4.0 + 8.0 + 12.0;
            var actual = b.ToArray();
            Assert.AreEqual(expected, actual);
        }

        [Test]
        public static void ReduceSumByColCpu()
        {
            var ctx = cpu;
            var input = new[,] { { 1.0, 2.0, 3.0, 4.0 }, { 5.0, 6.0, 7.0, 8.0 }, { 9.0, 10.0, 11.0, 12.0 } };
            var a = ctx.Allocate(input);
            a.Print();
            var b = ctx.Device.Allocate<double>(Shape.Create(3));
            ctx.Assign(b, ReduceSum(a, 1));
            b.Print();
            var expected = new double[input.GetLength(0)];
            expected[0] = 1.0 + 2.0 + 3.0 + 4.0;
            expected[1] = 5.0 + 6.0 + 7.0 + 8.0;
            expected[2] = 9.0 + 10.0 + 11.0 + 12.0;
            var actual = b.ToArray();
            Assert.AreEqual(expected, actual);
        }

        [Test]
        public static void ReduceSumByColGpu()
        {
            var ctx = gpu;
            var input = new[,] { { 1.0, 2.0, 3.0, 4.0 }, { 5.0, 6.0, 7.0, 8.0 }, { 9.0, 10.0, 11.0, 12.0 } };
            var a = ctx.Allocate(input);
            a.Print();
            var b = ctx.Device.Allocate<double>(Shape.Create(3));
            ctx.Assign(b, ReduceSum(a, 1));
            b.Print();
            var expected = new double[input.GetLength(0)];
            expected[0] = 1.0 + 2.0 + 3.0 + 4.0;
            expected[1] = 5.0 + 6.0 + 7.0 + 8.0;
            expected[2] = 9.0 + 10.0 + 11.0 + 12.0;
            var actual = b.ToArray();
            Assert.AreEqual(expected, actual);
        }

        [Test]
        public static void RepmatAsColViaLValueCpu()
        {
            var ctx = cpu;
            var input = new[] { 1.0, 2.0, 3.0 };
            var a = ctx.Allocate(input);
            a.Print();
            var b = ctx.Device.Allocate<double>(Shape.Create(3, 5));
            ctx.Assign(b, a.Reshape(-1, 1));
            b.Print();
            var expected = new[,]
            {
                {1.0, 1.0, 1.0, 1.0, 1.0},
                {2.0, 2.0, 2.0, 2.0, 2.0},
                {3.0, 3.0, 3.0, 3.0, 3.0}
            };
            var actual = b.ToArray2D();
            Assert.AreEqual(expected, actual);
        }

        [Test]
        public static void RepmatAsColViaLValueGpu()
        {
            var ctx = gpu;
            var input = new[] { 1.0, 2.0, 3.0 };
            var a = ctx.Allocate(input);
            a.Print();
            var b = ctx.Device.Allocate<double>(Shape.Create(3, 5));
            ctx.Assign(b, a.Reshape(-1, 1));
            b.Print();
            var expected = new[,]
            {
                {1.0, 1.0, 1.0, 1.0, 1.0},
                {2.0, 2.0, 2.0, 2.0, 2.0},
                {3.0, 3.0, 3.0, 3.0, 3.0}
            };
            var actual = b.ToArray2D();
            Assert.AreEqual(expected, actual);
        }

        [Test]
        public static void SimpleSoftmaxForwardCpu()
        {
            var ctx = cpu;
            var rng = new Random();
            const int M = 100;
            const int N = 10;
            var input = new double[M, N];
            for (var row = 0; row < M; ++row)
            {
                for (var col = 0; col < N; ++col)
                {
                    input[row, col] = rng.NextDouble();
                }
            }

            var a = ctx.Allocate(input);
            a.Print();
            var b = ctx.Device.Allocate<double>(Shape.Create(M, N));
            // we reduce sum the exp(a) on columns, and keepdims so it stays as a column vector.
            var softmax = Exp(a) / ReduceSum(Exp(a), true, 1);
            ctx.Assign(b, softmax);
            b.Print();

            var expected = new double[M, N];
            for (var row = 0; row < M; ++row)
            {
                var acc = Enumerable.Range(0, N).Select(col => Math.Exp(input[row, col])).Sum();
                for (var col = 0; col < N; ++col)
                {
                    expected[row, col] = Math.Exp(input[row, col]) / acc;
                }
            }
            var actual = b.ToArray2D();
            //TestUtil.AreClose(expected, actual, 1e-10);
            AreEqual(expected, actual);
        }

        [Test]
        public static void SimpleSoftmaxForwardGpu()
        {
            var ctx = gpu;
            var rng = new Random();
            const int M = 100;
            const int N = 10;
            var input = new double[M, N];
            for (var row = 0; row < M; ++row)
            {
                for (var col = 0; col < N; ++col)
                {
                    input[row, col] = rng.NextDouble();
                }
            }

            var a = ctx.Allocate(input);
            a.Print();
            var b = ctx.Device.Allocate<double>(Shape.Create(M, N));
            // we reduce sum the exp(a) on columns, and keepdims so it stays as a column vector.
            var softmax = Exp(a) / (ReduceSum(Exp(a), true, 1));
            ctx.Assign(b, softmax);
            b.Print();

            var expected = new double[M, N];
            for (var row = 0; row < M; ++row)
            {
                var acc = Enumerable.Range(0, N).Select(col => Math.Exp(input[row, col])).Sum();
                for (var col = 0; col < N; ++col)
                {
                    expected[row, col] = Math.Exp(input[row, col]) / acc;
                }
            }
            var actual = b.ToArray2D();
            AreClose(expected, actual, 1e-10);
            //TestUtil.AreEqual(expected, actual);
        }

        [Test]
        public static void TakeCpu()
        {
            var ctx = cpu;
            var source = new[,]
            {
                {0.0f, 0.1f, 0.2f},
                {1.0f, 1.1f, 1.2f},
                {2.0f, 2.1f, 2.2f},
                {3.0f, 3.1f, 3.2f},
                {4.0f, 4.1f, 4.2f},
            };
            var indices = new[,]
            {
                {0, 0, 1},
                {4, 1, 3}
            };
            var _source = ctx.Allocate(source);
            var _indices = ctx.Allocate(indices);
            var _embedded = ctx.Eval(Take(_indices, _source));
            ctx.Eval(_embedded.Reshape(6, -1)).Print();

            var expected = new[,]
            {
                {0.0f, 0.1f, 0.2f},
                {0.0f, 0.1f, 0.2f},
                {1.0f, 1.1f, 1.2f},
                {4.0f, 4.1f, 4.2f},
                {1.0f, 1.1f, 1.2f},
                {3.0f, 3.1f, 3.2f},
            };

            Assert.AreEqual(expected, ctx.Eval(_embedded.Reshape(6, -1)).ToArray2D());
        }

        [Test]
        public static void TakeGpu()
        {
            var ctx = gpu;
            var source = new[,]
            {
                {0.0f, 0.1f, 0.2f},
                {1.0f, 1.1f, 1.2f},
                {2.0f, 2.1f, 2.2f},
                {3.0f, 3.1f, 3.2f},
                {4.0f, 4.1f, 4.2f},
            };
            var indices = new[,]
            {
                {0, 0, 1},
                {4, 1, 3}
            };
            var _source = ctx.Allocate(source);
            var _indices = ctx.Allocate(indices);
            var _embedded = ctx.Eval(Take(_indices, _source));
            ctx.Eval(_embedded.Reshape(6, -1)).Print();

            var expected = new[,]
            {
                {0.0f, 0.1f, 0.2f},
                {0.0f, 0.1f, 0.2f},
                {1.0f, 1.1f, 1.2f},
                {4.0f, 4.1f, 4.2f},
                {1.0f, 1.1f, 1.2f},
                {3.0f, 3.1f, 3.2f},
            };

            Assert.AreEqual(expected, ctx.Eval(_embedded.Reshape(6, -1)).ToArray2D());
        }

        [Test]
        public static void TakeGradCpu()
        {
            var ctx = cpu;
            var gradout = new[,,]
            {
                {{0.0f, 0.1f, 0.2f}, {1.0f, 1.1f, 1.2f}, {2.0f, 2.1f, 2.2f}},
                {{3.0f, 3.1f, 3.2f}, {4.0f, 4.1f, 4.2f}, {5.0f, 5.1f, 5.2f}}
            };
            var indices = new[,]
            {
                {0, 0, 1},
                {4, 1, 3}
            };
            var _gradout = ctx.Allocate(gradout);
            var _indices = ctx.Allocate(indices);
            var _embedgrad = ctx.Eval(TakeGrad(_indices, _gradout, 5));
            _embedgrad.Print();

            var expected = new[,]
            {
                {0.0f + 1.0f, 0.1f + 1.1f, 0.2f + 1.2f},
                {2.0f + 4.0f, 2.1f + 4.1f, 2.2f + 4.2f},
                {0.0f, 0.0f, 0.0f},
                {5.0f, 5.1f, 5.2f},
                {3.0f, 3.1f, 3.2f}
            };

            Assert.AreEqual(expected, _embedgrad.ToArray2D());
        }

        [Test]
        public static void TakeGradGpu()
        {
            var ctx = gpu;
            var gradout = new[, ,]
            {
                {{0.0f, 0.1f, 0.2f}, {1.0f, 1.1f, 1.2f}, {2.0f, 2.1f, 2.2f}},
                {{3.0f, 3.1f, 3.2f}, {4.0f, 4.1f, 4.2f}, {5.0f, 5.1f, 5.2f}}
            };
            var indices = new[,]
            {
                {0, 0, 1},
                {4, 1, 3}
            };
            var _gradout = ctx.Allocate(gradout);
            var _indices = ctx.Allocate(indices);
            var _embedgrad = ctx.Eval(TakeGrad(_indices, _gradout, 5));
            _embedgrad.Print();

            var expected = new[,]
            {
                {0.0f + 1.0f, 0.1f + 1.1f, 0.2f + 1.2f},
                {2.0f + 4.0f, 2.1f + 4.1f, 2.2f + 4.2f},
                {0.0f, 0.0f, 0.0f},
                {5.0f, 5.1f, 5.2f},
                {3.0f, 3.1f, 3.2f}
            };

            Assert.AreEqual(expected, _embedgrad.ToArray2D());
        }

        [Test]
        public static void Slice1DCpu()
        {
            var ctx = gpu;
            var input = ctx.Allocate(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            ctx.Assign(input.Slice(), input.Slice() + 1.0);
            input.Print();
            Assert.AreEqual(new[] { 2.0, 3.0, 4.0, 5.0, 6.0 }, input.ToArray());

            ctx.Assign(input.Slice(1), input.Slice(1) + 1.0);
            input.Print();
            Assert.AreEqual(new[] { 2.0, 4.0, 4.0, 5.0, 6.0 }, input.ToArray());

            ctx.Assign(input.Slice(Range(1, 4)), input.Slice(Range(1, 4)) + 1.0);
            input.Print();
            Assert.AreEqual(new[] { 2.0, 5.0, 5.0, 6.0, 6.0 }, input.ToArray());
        }

        [Test]
        public static void Slice1DGpu()
        {
            var ctx = gpu;
            var input = ctx.Allocate(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            ctx.Assign(input.Slice(), input.Slice() + 1.0);
            input.Print();
            Assert.AreEqual(new[] { 2.0, 3.0, 4.0, 5.0, 6.0 }, input.ToArray());

            ctx.Assign(input.Slice(1), input.Slice(1) + 1.0);
            input.Print();
            Assert.AreEqual(new[] { 2.0, 4.0, 4.0, 5.0, 6.0 }, input.ToArray());

            ctx.Assign(input.Slice(Range(1, 4)), input.Slice(Range(1, 4)) + 1.0);
            input.Print();
            Assert.AreEqual(new[] { 2.0, 5.0, 5.0, 6.0, 6.0 }, input.ToArray());
        }

        [Test]
        public static void Slice2DCpu()
        {
            var ctx = cpu;
            var input = ctx.Allocate(new[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });

            ctx.Assign(input.Slice(), input.Slice() + 1.0);
            input.Print();
            Assert.AreEqual(new[,] { { 2.0, 3.0, 4.0 }, { 5.0, 6.0, 7.0 } }, input.ToArray2D());

            ctx.Assign(input.Slice(-1, Range(0, 2)), input.Slice(-1, Range(0, 2)) + 1.0);
            input.Print();
            Assert.AreEqual(new[,] { { 3.0, 4.0, 4.0 }, { 6.0, 7.0, 7.0 } }, input.ToArray2D());

            ctx.Assign(input.Slice(1, Range(0, 2)), input.Slice(1, Range(0, 2)) + 1.0);
            input.Print();
            Assert.AreEqual(new[,] { { 3.0, 4.0, 4.0 }, { 7.0, 8.0, 7.0 } }, input.ToArray2D());
        }

        [Test]
        public static void Slice2DGpu()
        {
            var ctx = gpu;
            var input = ctx.Allocate(new[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });

            ctx.Assign(input.Slice(), input.Slice() + 1.0);
            input.Print();
            Assert.AreEqual(new[,] { { 2.0, 3.0, 4.0 }, { 5.0, 6.0, 7.0 } }, input.ToArray2D());

            ctx.Assign(input.Slice(-1, Range(0, 2)), input.Slice(-1, Range(0, 2)) + 1.0);
            input.Print();
            Assert.AreEqual(new[,] { { 3.0, 4.0, 4.0 }, { 6.0, 7.0, 7.0 } }, input.ToArray2D());

            ctx.Assign(input.Slice(1, Range(0, 2)), input.Slice(1, Range(0, 2)) + 1.0);
            input.Print();
            Assert.AreEqual(new[,] { { 3.0, 4.0, 4.0 }, { 7.0, 8.0, 7.0 } }, input.ToArray2D());
        }

        [Test]
        public static void Slice3DCpu()
        {
            var ctx = cpu;
            var input = ctx.Allocate(new[, ,] { { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } }, { { 1.5, 2.5, 3.5 }, { 4.5, 5.5, 6.5 } } });

            ctx.Assign(input.Slice(), input.Slice() + 1.0);
            Assert.AreEqual(new[, ,] { { { 2.0, 3.0, 4.0 }, { 5.0, 6.0, 7.0 } }, { { 2.5, 3.5, 4.5 }, { 5.5, 6.5, 7.5 } } }, input.ToArray3D());

            ctx.Assign(input.Slice(-1, -1, Range(0, 2)), input.Slice(-1, -1, Range(0, 2)) + 1.0);
            Assert.AreEqual(new[, ,] { { { 3.0, 4.0, 4.0 }, { 6.0, 7.0, 7.0 } }, { { 3.5, 4.5, 4.5 }, { 6.5, 7.5, 7.5 } } }, input.ToArray3D());

            ctx.Assign(input.Slice(1, -1, Range(0, 2)), input.Slice(1, -1, Range(0, 2)) + 1.0);
            Assert.AreEqual(new[, ,]
            {
                {
                    { 3.0, 4.0, 4.0 },
                    { 6.0, 7.0, 7.0 }
                },
                {
                    { 4.5, 5.5, 4.5 },
                    { 7.5, 8.5, 7.5 }
                }
            }, input.ToArray3D());
        }

        [Test]
        public static void Slice3DGpu()
        {
            var ctx = gpu;
            var input = ctx.Allocate(new[, ,] { { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } }, { { 1.5, 2.5, 3.5 }, { 4.5, 5.5, 6.5 } } });

            ctx.Assign(input.Slice(), input.Slice() + 1.0);
            Assert.AreEqual(new[, ,] { { { 2.0, 3.0, 4.0 }, { 5.0, 6.0, 7.0 } }, { { 2.5, 3.5, 4.5 }, { 5.5, 6.5, 7.5 } } }, input.ToArray3D());

            ctx.Assign(input.Slice(-1, -1, Range(0, 2)), input.Slice(-1, -1, Range(0, 2)) + 1.0);
            Assert.AreEqual(new[, ,] { { { 3.0, 4.0, 4.0 }, { 6.0, 7.0, 7.0 } }, { { 3.5, 4.5, 4.5 }, { 6.5, 7.5, 7.5 } } }, input.ToArray3D());

            ctx.Assign(input.Slice(1, -1, Range(0, 2)), input.Slice(1, -1, Range(0, 2)) + 1.0);
            Assert.AreEqual(new[, ,]
            {
                {
                    { 3.0, 4.0, 4.0 },
                    { 6.0, 7.0, 7.0 }
                },
                {
                    { 4.5, 5.5, 4.5 },
                    { 7.5, 8.5, 7.5 }
                }
            }, input.ToArray3D());
        }

        [Test]
        public static void RandomNormalCpu()
        {
            var ctx = cpu;

            //var data = ctx.Eval(RandomNormal<double>(Shape.Create(100, 100), seed: 0UL));
            //data.Print();

            //var mean = ctx.Eval(ReduceMean(data));
            //mean.Print();

            //Assert.That(mean.ToScalar(), Is.EqualTo(0.0).Within(1e-2));
        }

        [Test]
        public static void RandomNormalGpu()
        {
            var ctx = gpu;

            var data = ctx.Eval(RandomNormal<double>(Shape.Create(100, 100), seed: 0UL));
            data.Print();

            var mean = ctx.Eval(ReduceMean(data));
            mean.Print();

            Assert.That(mean.ToScalar(), Is.EqualTo(0.0).Within(1e-2));
        }

        [Test]
        public static void DropoutForwardGpu()
        {
            var ctx = gpu;

            var data = ctx.Allocate(new[] {1.0f, 2.0f, 3.0f, 4.0f});
            var mask = ctx.Allocate(new[] {6U, 3U, 9U, 2U});
            var threshold = 5U;
            var scale = 2.0;
            var result = ctx.Eval(Dropout(data, mask, threshold, scale));
            result.Print();
        }

        [Test]
        public static void RandomUniformGpu()
        {
            var ctx = gpu;

            var data = ctx.Eval((2.0f.AsScalar() * RandomUniform<float>(Shape.Create(10, 10)) - 1.0f.AsScalar()) * 5.0f.AsScalar());
            data.Print();
        }
    }
}
