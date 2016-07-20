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

            var rng = new Random(0);

            var mfr = new MatFileReader("../tests/AleaTKTest/data/lstm_small.mat");

            var inputSize = mfr.GetInt("InputSize");
            var seqLength = mfr.GetInt("SeqLength");
            var hiddenSize = mfr.GetInt("HiddenSize");
            var batchSize = mfr.GetInt("BatchSize");

            var x = Variable<float>(PartialShape.Create(seqLength, batchSize, inputSize));
            var lstm = new LSTM<float>(x, hiddenSize);

            var ctx = Context.GpuContext(0);
            var exe = new Executor(ctx, lstm.Y);

            exe.Initalize();

<<<<<<< 48ffa75a6d00ddbc17f3e7bf64bfec7edd6ceff2
            var input = mfr.GetDoubleArray("X").Select(n => (float) n).ToArray();
=======
            exe.AssignTensor(lstm.CX, Fill(Shape.Create(batchSize, hiddenSize), 0.0f));
            exe.AssignTensor(lstm.HX, Fill(Shape.Create(batchSize, hiddenSize), 0.0f));

            //var input = new float[seqLength, batchSize, inputSize];
            //RandomMat(input, rng);

            var input_ = /*[shape  (3, 2, 5) ]*/ new []{ -0.49803245069230, 1.92953205381699, 0.94942080692576, 0.08755124138519, -1.22543551883017, 0.84436297640155, -1.00021534738956, -1.54477109677761, 1.18802979235230, 0.31694261192485, 0.92085882378082, 0.31872765294302, 0.85683061190269, -0.65102559330015, -1.03424284178446, 0.68159451828163, -0.80340966417384, -0.68954977775020, -0.45553250351734, 0.01747915902506, -0.35399391125348, -1.37495129341802, -0.64361840283289, -2.22340315222443, 0.62523145102719, -1.60205765560675, -1.10438333942845, 0.05216507926097, -0.73956299639131, 1.54301459540674};
            var input = input_.Select(n => (float) n).ToArray();


>>>>>>> 89707ada1ea7cf18d1d9ec732073f0aec84dd67d
            Context.CpuContext.Eval(input.AsTensor().Reshape(seqLength*batchSize, inputSize)).Print();

            exe.AssignTensor(x, input.AsTensor(Shape.Create(seqLength, batchSize, inputSize)));

            var w = mfr.GetDoubleArray("WLSTM").Select(n => (float) n).ToArray();
            exe.AssignTensor(lstm.W, w.AsTensor(Shape.Create(inputSize + hiddenSize + 1, 4*hiddenSize)));

            exe.Forward();

            var H = mfr.GetDoubleArray("H");
            H.AsTensor(Shape.Create(seqLength*batchSize, hiddenSize)).Print();

            ctx.Eval(exe.GetTensor(lstm.Y).Reshape(seqLength * batchSize, -1)).Print();

            ctx.ToGpuContext().Stream.Synchronize();
        }
    }
}
