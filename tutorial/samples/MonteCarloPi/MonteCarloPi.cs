using System;
using NUnit.Framework;

using Alea;
using Alea.Parallel;
using AleaTK;
using Context = AleaTK.Context;
using static AleaTK.Library;

namespace Tutorial.Samples
{
    public class MonteCarloPi
    {
        private static void Main()
        {
            PiEstimationGpu();
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
                ctx.Assign(points, RandomUniform<double2>(seed, offset));
                ctx.Assign(pi, i == 0 ? ReduceMean(pis) : (pi + ReduceMean(pis)) / 2.0);
            }

            Console.WriteLine($"Pi = {pi.ToScalar()}");
            Assert.That(pi.ToScalar(), Is.EqualTo(Math.PI).Within(error));
        }

        [Test]
        public static void PiEstimationGpu()
        {
            EstimatePi(Context.GpuContext(0), 100, 10000000, 1e-3);
        }

        [Test]
        public static void PiEstimationCpu()
        {
            EstimatePi(Context.CpuContext, 5, 100000, 1e-2);
        }
    }
}
