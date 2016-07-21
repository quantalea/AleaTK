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

        public static void LSTMvsPython()
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
        public static void LSTMvsCUDNN()
        {
            var ctx = Context.GpuContext(0);
            var inputSize = 5;
            var seqLength = 3;
            var batchSize = 2;
            var hiddenSize = 4;

            var data = 
                Context.CpuContext.Eval((2.0f.AsScalar()*
                                         RandomUniform<float>(Shape.Create(seqLength, batchSize, inputSize)) -
                                         1.0f.AsScalar())).ToArray3D();
            data.AsTensor(Shape.Create(seqLength*batchSize, inputSize)).Print();

            //var h0 = Context.CpuContext.Eval(Fill(Shape.Create(batchSize, hiddenSize), 0.0f)).ToArray2D();
            //var c0 = Context.CpuContext.Eval(Fill(Shape.Create(batchSize, hiddenSize), 0.0f)).ToArray2D();
            var h0 = Context.CpuContext.Eval(RandomNormal<float>(Shape.Create(batchSize, hiddenSize))).ToArray2D();
            var c0 = Context.CpuContext.Eval(RandomNormal<float>(Shape.Create(batchSize, hiddenSize))).ToArray2D();

            var dy =
                Context.CpuContext.Eval((2.0f.AsScalar() *
                                         RandomUniform<float>(Shape.Create(seqLength, batchSize, hiddenSize)) -
                                         1.0f.AsScalar())).ToArray3D();
            dy.AsTensor(Shape.Create(seqLength * batchSize, hiddenSize)).Print();

            var wi = 0.5f;
            var wf = 0.4f;
            var wo = 0.3f;
            var wa = 0.2f;
            var ui = 0.5f;
            var uf = 0.4f;
            var uo = 0.3f;
            var ua = 0.1f;
            var bi = 0.5f;
            var bf = 0.4f;
            var bo = 0.3f;
            var ba = 0.2f;
            
            float[,,] y1, y2, dx1, dx2;
            float[,] dcx1, dcx2, dhx1, dhx2;
            float[,] dw1, dw2;
            {
                // calc with cuDNN
                var x = Variable<float>(PartialShape.Create(seqLength, batchSize, inputSize));
                var lstm = new RNN<float>(x, 1, hiddenSize);
                var exe = new Executor(ctx, lstm.Y);
                exe.Initalize();

                // set input
                exe.AssignTensor(lstm.X, data.AsTensor());

                // set states
                exe.AssignTensor(lstm.CX, c0.AsTensor(Shape.Create(1, batchSize, hiddenSize)));
                exe.AssignTensor(lstm.HX, h0.AsTensor(Shape.Create(1, batchSize, hiddenSize)));

                // set weigths
                // cuDNN matrices order: IFAO
                var w = exe.GetTensor(lstm.W).Reshape(inputSize * 4 + hiddenSize * 4 + 2 * 4, hiddenSize);
                var offset = 0;
                // Wi
                ctx.Assign(w.Slice(Range(offset, offset + inputSize)), Fill(Shape.Create(inputSize, hiddenSize), wi));
                offset += inputSize;
                // Wf
                ctx.Assign(w.Slice(Range(offset, offset + inputSize)), Fill(Shape.Create(inputSize, hiddenSize), wf));
                offset += inputSize;
                // Wa
                ctx.Assign(w.Slice(Range(offset, offset + inputSize)), Fill(Shape.Create(inputSize, hiddenSize), wa));
                offset += inputSize;
                // Wo
                ctx.Assign(w.Slice(Range(offset, offset + inputSize)), Fill(Shape.Create(inputSize, hiddenSize), wo));
                offset += inputSize;
                // Ui
                ctx.Assign(w.Slice(Range(offset, offset + hiddenSize)), Fill(Shape.Create(hiddenSize, hiddenSize), ui));
                offset += hiddenSize;
                // Uf
                ctx.Assign(w.Slice(Range(offset, offset + hiddenSize)), Fill(Shape.Create(hiddenSize, hiddenSize), uf));
                offset += hiddenSize;
                // Ua
                ctx.Assign(w.Slice(Range(offset, offset + hiddenSize)), Fill(Shape.Create(hiddenSize, hiddenSize), ua));
                offset += hiddenSize;
                // Uo
                ctx.Assign(w.Slice(Range(offset, offset + hiddenSize)), Fill(Shape.Create(hiddenSize, hiddenSize), uo));
                offset += hiddenSize;
                // Bi
                ctx.Assign(w.Slice(offset), Fill(Shape.Create(1, hiddenSize), bi));
                offset++;
                // Bf
                ctx.Assign(w.Slice(offset), Fill(Shape.Create(1, hiddenSize), bf));
                offset++;
                // Ba
                ctx.Assign(w.Slice(offset), Fill(Shape.Create(1, hiddenSize), ba));
                offset++;
                // Bo
                ctx.Assign(w.Slice(offset), Fill(Shape.Create(1, hiddenSize), bo));

                exe.Forward();

                y1 = exe.GetTensor(lstm.Y).ToArray3D();

                exe.AssignGradientDirectly(lstm.Y, dy.AsTensor());

                exe.Backward();

                dx1 = exe.GetGradient(lstm.X).ToArray3D();
                dcx1 = exe.GetGradient(lstm.CX).Reshape(batchSize, hiddenSize).ToArray2D();
                dhx1 = exe.GetGradient(lstm.HX).Reshape(batchSize, hiddenSize).ToArray2D();

                // we make dw follow the shape as (1 + inputSize + hiddenSize, 4*hiddenSize)
                // also cuDNN is fortran order, so we need transpose it
                var dwCUDNN = exe.GetGradient(lstm.W).ToArray().AsTensor();
                dw1 = new float[1 + inputSize + hiddenSize, 4*hiddenSize];
                var dw1Tensor = Reference<float>(dw1);
                var cpu = Context.CpuContext;
                offset = 0;
                // cuDNN order: IFAO , also need transpose, cuDNN stores data in fortran order.
                // Wi
                cpu.Assign(dw1Tensor.Slice(Range(1, inputSize + 1), Range(0, hiddenSize)), dwCUDNN.Slice(Range(offset, offset + inputSize * hiddenSize)).Reshape(hiddenSize, inputSize).T);
                offset += inputSize*hiddenSize;
                // Wf
                cpu.Assign(dw1Tensor.Slice(Range(1, inputSize + 1), Range(hiddenSize, 2 * hiddenSize)), dwCUDNN.Slice(Range(offset, offset + inputSize * hiddenSize)).Reshape(hiddenSize, inputSize).T);
                offset += inputSize * hiddenSize;
                // Wa
                cpu.Assign(dw1Tensor.Slice(Range(1, inputSize + 1), Range(3 * hiddenSize, 4 * hiddenSize)), dwCUDNN.Slice(Range(offset, offset + inputSize * hiddenSize)).Reshape(hiddenSize, inputSize).T);
                offset += inputSize * hiddenSize;
                // Wo
                cpu.Assign(dw1Tensor.Slice(Range(1, inputSize + 1), Range(2 * hiddenSize, 3 * hiddenSize)), dwCUDNN.Slice(Range(offset, offset + inputSize * hiddenSize)).Reshape(hiddenSize, inputSize).T);
                offset += inputSize * hiddenSize;
                // Ui
                cpu.Assign(dw1Tensor.Slice(Range(inputSize + 1, -1), Range(0, hiddenSize)), dwCUDNN.Slice(Range(offset, offset + hiddenSize * hiddenSize)).Reshape(hiddenSize, hiddenSize).T);
                offset += hiddenSize * hiddenSize;
                // Uf
                cpu.Assign(dw1Tensor.Slice(Range(inputSize + 1, -1), Range(hiddenSize, 2 * hiddenSize)), dwCUDNN.Slice(Range(offset, offset + hiddenSize * hiddenSize)).Reshape(hiddenSize, hiddenSize).T);
                offset += hiddenSize * hiddenSize;
                // Ua
                cpu.Assign(dw1Tensor.Slice(Range(inputSize + 1, -1), Range(3 * hiddenSize, 4 * hiddenSize)), dwCUDNN.Slice(Range(offset, offset + hiddenSize * hiddenSize)).Reshape(hiddenSize, hiddenSize).T);
                offset += hiddenSize * hiddenSize;
                // Uo
                cpu.Assign(dw1Tensor.Slice(Range(inputSize + 1, -1), Range(2 * hiddenSize, 3 * hiddenSize)), dwCUDNN.Slice(Range(offset, offset + hiddenSize * hiddenSize)).Reshape(hiddenSize, hiddenSize).T);
                offset += hiddenSize * hiddenSize;
                // Bi
                cpu.Assign(dw1Tensor.Slice(0, Range(0, hiddenSize)), dwCUDNN.Slice(Range(offset, offset + hiddenSize)).Reshape(hiddenSize, 1).T);
                offset += hiddenSize;
                // Bf
                cpu.Assign(dw1Tensor.Slice(0, Range(hiddenSize, 2 * hiddenSize)), dwCUDNN.Slice(Range(offset, offset + hiddenSize)).Reshape(hiddenSize, 1).T);
                offset += hiddenSize;
                // Ba
                cpu.Assign(dw1Tensor.Slice(0, Range(3 * hiddenSize, 4 * hiddenSize)), dwCUDNN.Slice(Range(offset, offset + hiddenSize)).Reshape(hiddenSize, 1).T);
                offset += hiddenSize;
                // Bo
                cpu.Assign(dw1Tensor.Slice(0, Range(2 * hiddenSize, 3 * hiddenSize)), dwCUDNN.Slice(Range(offset, offset + hiddenSize)).Reshape(hiddenSize, 1).T);
            }

            {
                // calc with lstm
                var x = Variable<float>(PartialShape.Create(seqLength, batchSize, inputSize));
                var lstm = new LSTM<float>(x, hiddenSize, forgetBiasInit: 0.0);
                var exe = new Executor(ctx, lstm.Y);
                exe.Initalize();

                // set input
                exe.AssignTensor(lstm.X, data.AsTensor());

                // set states
                exe.AssignTensor(lstm.CX, c0.AsTensor());
                exe.AssignTensor(lstm.HX, h0.AsTensor());

                // set weights
                var w = exe.GetTensor(lstm.W);
                // Wi
                ctx.Assign(w.Slice(Range(1, inputSize + 1), Range(0, hiddenSize)), Fill(Shape.Create(inputSize, hiddenSize), wi));
                // Wf
                ctx.Assign(w.Slice(Range(1, inputSize + 1), Range(hiddenSize, 2*hiddenSize)), Fill(Shape.Create(inputSize, hiddenSize), wf));
                // Wo
                ctx.Assign(w.Slice(Range(1, inputSize + 1), Range(2*hiddenSize, 3*hiddenSize)), Fill(Shape.Create(inputSize, hiddenSize), wo));
                // Wa
                ctx.Assign(w.Slice(Range(1, inputSize + 1), Range(3*hiddenSize, 4*hiddenSize)), Fill(Shape.Create(inputSize, hiddenSize), wa));
                // Ui
                ctx.Assign(w.Slice(Range(inputSize + 1, -1), Range(0, hiddenSize)), Fill(Shape.Create(hiddenSize, hiddenSize), ui));
                // Uf
                ctx.Assign(w.Slice(Range(inputSize + 1, -1), Range(hiddenSize, 2*hiddenSize)), Fill(Shape.Create(hiddenSize, hiddenSize), uf));
                // Uo
                ctx.Assign(w.Slice(Range(inputSize + 1, -1), Range(2*hiddenSize, 3*hiddenSize)), Fill(Shape.Create(hiddenSize, hiddenSize), uo));
                // Ua
                ctx.Assign(w.Slice(Range(inputSize + 1, -1), Range(3*hiddenSize, 4*hiddenSize)), Fill(Shape.Create(hiddenSize, hiddenSize), ua));
                // Bi
                ctx.Assign(w.Slice(0, Range(0, hiddenSize)), Fill(Shape.Create(1, hiddenSize), bi));
                // Bf
                ctx.Assign(w.Slice(0, Range(hiddenSize, 2*hiddenSize)), Fill(Shape.Create(1, hiddenSize), bf));
                // Bo
                ctx.Assign(w.Slice(0, Range(2*hiddenSize, 3*hiddenSize)), Fill(Shape.Create(1, hiddenSize), bo));
                // Ba
                ctx.Assign(w.Slice(0, Range(3*hiddenSize, 4*hiddenSize)), Fill(Shape.Create(1, hiddenSize), ba));

                exe.Forward();

                y2 = exe.GetTensor(lstm.Y).ToArray3D();

                exe.AssignGradientDirectly(lstm.Y, dy.AsTensor());

                exe.Backward();

                dx2 = exe.GetGradient(lstm.X).ToArray3D();
                dcx2 = exe.GetGradient(lstm.CX).Reshape(batchSize, hiddenSize).ToArray2D();
                dhx2 = exe.GetGradient(lstm.HX).Reshape(batchSize, hiddenSize).ToArray2D();
                dw2 = exe.GetGradient(lstm.W).ToArray2D();
            }

            y1.AsTensor(Shape.Create(seqLength*batchSize, hiddenSize)).Print();
            y2.AsTensor(Shape.Create(seqLength*batchSize, hiddenSize)).Print();
            AreClose(y1.AsTensor().ToArray(), y2.AsTensor().ToArray(), 1e-6);

            dx1.AsTensor(Shape.Create(seqLength * batchSize, inputSize)).Print();
            dx2.AsTensor(Shape.Create(seqLength * batchSize, inputSize)).Print();
            AreClose(dx1.AsTensor().ToArray(), dx2.AsTensor().ToArray(), 1e-6);

            dcx1.AsTensor().Print();
            dcx2.AsTensor().Print();
            AreClose(dcx1, dcx2, 1e-6);

            dhx1.AsTensor().Print();
            dhx2.AsTensor().Print();
            AreClose(dhx1, dhx2, 1e-6);

            dw1.AsTensor().Print();
            dw2.AsTensor().Print();
            AreClose(dw1, dw2, 1e-6);
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
