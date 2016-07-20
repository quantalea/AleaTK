using System;
using System.Linq;
using AleaTK;
using AleaTK.ML;
using AleaTK.ML.Operator;
using csmatio.io;
using csmatio.types;
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

        private static void RandomMat(float[,,] mat, Random rng)
        {
            for (var i = 0; i < mat.GetLength(0); ++i)
            {
                for (var j = 0; j < mat.GetLength(1); ++j)
                {
                    for (var k = 0; k < mat.GetLength(2); ++k)
                    {
                        mat[i, j, k] = (float) rng.NextDouble();
                    }
                }
            }
        }

        [Test]
        public static void TestLSTM()
        {
            var mfr = new MatFileReader(@"../tests/AleaTKTest/data/lstm_small.mat");

            var inputSize = mfr.GetInt("InputSize");
            var seqLength = mfr.GetInt("SeqLength");
            var hiddenSize = mfr.GetInt("HiddenSize");
            var batchSize = mfr.GetInt("BatchSize");

            var x = Variable<float>(PartialShape.Create(seqLength, batchSize, inputSize));
            var lstm = new LSTM<float>(x, hiddenSize);

            var ctx = Context.GpuContext(0);
            var exe = new Executor(ctx, lstm.Y);

            exe.Initalize();

            var h0 = mfr.GetDoubleArray("h0").Select(n => (float)n).ToArray();
            var c0 = mfr.GetDoubleArray("c0").Select(n => (float)n).ToArray();
            exe.AssignTensor(lstm.CX, c0.AsTensor(Shape.Create(batchSize, hiddenSize)));
            exe.AssignTensor(lstm.HX, h0.AsTensor(Shape.Create(batchSize, hiddenSize)));

            var input = mfr.GetDoubleArray("X").Select(n => (float) n).ToArray();
            //input.AsTensor(Shape.Create(seqLength*batchSize, inputSize)).Print();
            exe.AssignTensor(x, input.AsTensor(Shape.Create(seqLength, batchSize, inputSize)));

            var w = mfr.GetDoubleArray("W").Select(n => (float)n).ToArray();
            w.AsTensor(Shape.Create(inputSize + hiddenSize + 1, 4 * hiddenSize)).Print();
            exe.AssignTensor(lstm.W, w.AsTensor(Shape.Create(inputSize + hiddenSize + 1, 4 * hiddenSize)));

            exe.Forward();

            var H = mfr.GetDoubleArray("H").Select(n => (float)n).ToArray();
            H.AsTensor(Shape.Create(seqLength*batchSize, hiddenSize)).Print();

            var myH = exe.GetTensor(lstm.Y).ToArray();
            myH.AsTensor(Shape.Create(seqLength*batchSize, hiddenSize)).Print();

            AreClose(H, myH, 1e-6);

            var dH = mfr.GetDoubleArray("dH").Select(n => (float) n).ToArray();
            exe.AssignGradientDirectly(lstm.Y, dH.AsTensor(Shape.Create(seqLength, batchSize, hiddenSize)));

            exe.Backward();

            var dX = mfr.GetDoubleArray("dX").Select(n => (float) n).ToArray();
            dX.AsTensor(Shape.Create(seqLength*batchSize, inputSize)).Print();

            var dXmy = exe.GetGradient(lstm.X).ToArray();
            dXmy.AsTensor(Shape.Create(seqLength * batchSize, inputSize)).Print();
            AreClose(dX, dXmy, 1e-6);

            var dW = mfr.GetDoubleArray("dW").Select(n => (float)n).ToArray();
            dW.AsTensor(Shape.Create(inputSize + hiddenSize + 1, 4*hiddenSize)).Print();

            var dWmy = exe.GetGradient(lstm.W).ToArray();
            dWmy.AsTensor(Shape.Create(lstm.W.Shape.AsArray)).Print();
            AreClose(dW, dWmy, 1e-6);

            var dc0 = mfr.GetDoubleArray("dc0").Select(n => (float)n).ToArray();
            dc0.AsTensor(Shape.Create(batchSize, hiddenSize)).Print();

            var dc0my = exe.GetGradient(lstm.CX).ToArray();
            dc0my.AsTensor(Shape.Create(batchSize, hiddenSize)).Print();
            AreClose(dc0, dc0my, 1e-6);

            var dh0 = mfr.GetDoubleArray("dh0").Select(n => (float)n).ToArray();
            dh0.AsTensor(Shape.Create(batchSize, hiddenSize)).Print();

            var dh0my = exe.GetGradient(lstm.HX).ToArray();
            dh0my.AsTensor(Shape.Create(batchSize, hiddenSize)).Print();
            AreClose(dh0, dh0my, 1e-6);

            ctx.ToGpuContext().Stream.Synchronize();
        }

        [Test]
        public static void TestLSTMWithDropout()
        {
            var mfr = new MatFileReader(@"../tests/AleaTKTest/data/lstm_small.mat");

            var inputSize = mfr.GetInt("InputSize");
            var seqLength = mfr.GetInt("SeqLength");
            var hiddenSize = mfr.GetInt("HiddenSize");
            var batchSize = mfr.GetInt("BatchSize");

            var x = Variable<float>(PartialShape.Create(seqLength, batchSize, inputSize));
            var dropout0 = new Dropout<float>(x, dropoutProb: 0.3);
            var lstm1 = new LSTM<float>(dropout0.Output, hiddenSize);
            var dropout1 = new Dropout<float>(lstm1.Y, dropoutProb: 0.5);
            var lstm2 = new LSTM<float>(dropout1.Output, hiddenSize);
            var dropout2 = new Dropout<float>(lstm2.Y, dropoutProb: 0.6);

            var ctx = Context.GpuContext(0);
            var exe = new Executor(ctx, dropout2.Output);

            exe.Initalize();

            var h0 = mfr.GetDoubleArray("h0").Select(n => (float)n).ToArray();
            var c0 = mfr.GetDoubleArray("c0").Select(n => (float)n).ToArray();
            exe.AssignTensor(lstm1.CX, c0.AsTensor(Shape.Create(batchSize, hiddenSize)));
            exe.AssignTensor(lstm1.HX, h0.AsTensor(Shape.Create(batchSize, hiddenSize)));

            exe.AssignTensor(lstm2.CX, Fill(Shape.Create(batchSize, hiddenSize), 0.0f));
            exe.AssignTensor(lstm2.HX, Fill(Shape.Create(batchSize, hiddenSize), 0.0f));

            var input = mfr.GetDoubleArray("X").Select(n => (float)n).ToArray();
            ////input.AsTensor(Shape.Create(seqLength*batchSize, inputSize)).Print();
            exe.AssignTensor(x, input.AsTensor(Shape.Create(seqLength, batchSize, inputSize)));

            //var w = mfr.GetDoubleArray("W").Select(n => (float)n).ToArray();
            ////w.AsTensor(Shape.Create(inputSize + hiddenSize + 1, 4 * hiddenSize)).Print();
            //exe.AssignTensor(lstm.W, w.AsTensor(Shape.Create(inputSize + hiddenSize + 1, 4 * hiddenSize)));

            exe.Forward();

            ////var H = mfr.GetDoubleArray("H").Select(n => (float)n).ToArray();
            ////H.AsTensor(Shape.Create(seqLength * batchSize, hiddenSize)).Print();

            var myH = exe.GetTensor(dropout2.Output).ToArray();
            myH.AsTensor(Shape.Create(seqLength * batchSize, hiddenSize)).Print();

            //var myH2 = exe.GetTensor(dropout.Output).ToArray();
            //myH2.AsTensor(Shape.Create(seqLength * batchSize, hiddenSize)).Print();

            ////AreClose(H, myH, 1e-6);

            var dH = mfr.GetDoubleArray("dH").Select(n => (float)n).ToArray();
            exe.AssignGradientDirectly(dropout2.Output, dH.AsTensor(Shape.Create(seqLength, batchSize, hiddenSize)));

            exe.Backward();

            ////var dX = mfr.GetDoubleArray("dX").Select(n => (float)n).ToArray();
            ////dX.AsTensor(Shape.Create(seqLength * batchSize, inputSize)).Print();

            //var dDropout = exe.GetGradient(lstm.Y).ToArray();
            //dDropout.AsTensor(Shape.Create(seqLength*batchSize, hiddenSize)).Print();

            var dXmy = exe.GetTensor(dropout0.Output).ToArray();
            dXmy.AsTensor(Shape.Create(seqLength * batchSize, inputSize)).Print();
            //AreClose(dX, dXmy, 1e-6);

            //var dW = mfr.GetDoubleArray("dW").Select(n => (float)n).ToArray();
            //dW.AsTensor(Shape.Create(inputSize + hiddenSize + 1, 4 * hiddenSize)).Print();

            //var dWmy = exe.GetGradient(lstm.W).ToArray();
            //dWmy.AsTensor(Shape.Create(lstm.W.Shape.AsArray)).Print();
            //AreClose(dW, dWmy, 1e-6);

            //var dc0 = mfr.GetDoubleArray("dc0").Select(n => (float)n).ToArray();
            //dc0.AsTensor(Shape.Create(batchSize, hiddenSize)).Print();

            //var dc0my = exe.GetGradient(lstm.CX).ToArray();
            //dc0my.AsTensor(Shape.Create(batchSize, hiddenSize)).Print();
            //AreClose(dc0, dc0my, 1e-6);

            //var dh0 = mfr.GetDoubleArray("dh0").Select(n => (float)n).ToArray();
            //dh0.AsTensor(Shape.Create(batchSize, hiddenSize)).Print();

            //var dh0my = exe.GetGradient(lstm.HX).ToArray();
            //dh0my.AsTensor(Shape.Create(batchSize, hiddenSize)).Print();
            //AreClose(dh0, dh0my, 1e-6);

            ctx.ToGpuContext().Stream.Synchronize();
        }
    }
}
