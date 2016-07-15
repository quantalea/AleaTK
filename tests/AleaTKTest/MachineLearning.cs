using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.Remoting.Messaging;
using System.Text;
using System.Threading.Tasks;
using AleaTK;
using AleaTK.ML;
using AleaTK.ML.Operator;
using NUnit.Framework;
using static AleaTK.Library;
using static AleaTK.ML.Library;
using static AleaTKTest.Common;

namespace AleaTKTest
{
    public static class MachineLearning
    {
        [Test]
        public static void SimpleLogisticRegression()
        {
            //const int N = 8;
            //const int D = 5;
            //const int P = 3;
            //const double learn = 0.001;

            const int N = 100;
            const int D = 784;
            const int P = 10;
            const double learn = 0.00005;

            var input = Variable<double>();
            var label = Variable<double>();
            var weights = Parameter(0.01 * RandomUniform<double>(Shape.Create(D, P)));
            var pred = Dot(input, weights);
            var loss = L2Loss(pred, label);

            var ctx = Context.GpuContext(0);
            var opt = new GradientDescentOptimizer(ctx, loss, learn);

            // set some data
            var rng = new Random(42);
            var _input = RandMat(rng, N, D);
            var _label = Dot(_input, RandMat(rng, D, P)).Add(RandMat(rng, N, P).Mul(0.1));
            opt.AssignTensor(input, _input.AsTensor());
            opt.AssignTensor(label, _label.AsTensor());

            opt.Initalize();
            for (var i = 0; i < 800; ++i)
            {
                opt.Forward();
                opt.Backward();
                opt.Optimize();
                if (i % 20 == 0)
                {
                    Console.WriteLine($"loss = {opt.GetTensor(loss).ToScalar()}");
                }
            }
        }

        [Test]
        public static void TestRNN()
        {
            const int miniBatch = 64;
            const int seqLength = 20;
            const int numLayers = 2;
            const int hiddenSize = 512;
            const int inputSize = hiddenSize;

            var x = Variable<float>(PartialShape.Create(miniBatch, seqLength, inputSize));
            //var x = Variable<float>(PartialShape.Create(miniBatch, inputSize, seqLength));
            var rnn = new RNN<float>(x, numLayers, hiddenSize);

            var ctx = Context.GpuContext(0);
            var opt = new GradientDescentOptimizer(ctx, rnn.Y, 0.001);
            opt.Initalize();

            opt.AssignTensor(x, Fill(Shape.Create(miniBatch, seqLength, inputSize), 1.0f));
            //opt.AssignTensor(x, Fill(Shape.Create(miniBatch, inputSize, seqLength), 1.0f));

            opt.Forward();
            opt.Backward();
            ctx.ToGpuContext().Stream.Synchronize();
        }
    }
}
