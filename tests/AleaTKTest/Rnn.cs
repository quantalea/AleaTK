using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using AleaTK;
using AleaTK.ML;
using AleaTK.ML.Operator;
using AleaTKUtil;
using static AleaTK.Library;
using static AleaTK.ML.Library;
using static AleaTKUtil.Common;
using Context = AleaTK.Context;
using NUnit.Framework;

namespace AleaTKTest
{
    public static class Rnn
    {
        public static void SetWeights(AleaTK.Context ctx, Tensor<float> w, int inputSize, int hiddenSize)
        {
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
        }

        [Test]
        public static void RnnAgainstRnnDynamic()
        {
            var ctx = Context.GpuContext(0);
            var inputSize = 5;
            var seqLength = 3;
            var batchSize = 2;
            var hiddenSize = 4;
            var error = 1e-5;

            var data = Context.CpuContext.Eval(RandomUniform<float>(-1, 1, Shape.Create(seqLength, batchSize, inputSize))).ToArray3D();
            data.AsTensor(Shape.Create(seqLength*batchSize, inputSize)).Print();

            var h0 = Context.CpuContext.Eval(RandomNormal<float>(Shape.Create(batchSize, hiddenSize))).ToArray2D();
            var c0 = Context.CpuContext.Eval(RandomNormal<float>(Shape.Create(batchSize, hiddenSize))).ToArray2D();
            var dy = Context.CpuContext.Eval(RandomUniform<float>(-1, 1, Shape.Create(seqLength, batchSize, hiddenSize))).ToArray3D();

            float[,,] y1, y2, dx1, dx2;
            float[,] cy1, cy2, hy1, hy2;
            float[,] dcx1, dcx2, dhx1, dhx2;
            float[] dw1, dw2;

            {
                var x = Variable<float>(PartialShape.Create(seqLength, batchSize, inputSize));
                var lstm = new Rnn<float>(new LstmRnnType(), x, 1, hiddenSize, dropout: 0.0);
                var exe = new Executor(ctx, lstm.Y);
                exe.Initalize();

                // set input
                exe.AssignTensor(lstm.X, data.AsTensor());

                // set states
                exe.AssignTensor(lstm.CX, c0.AsTensor(Shape.Create(1, batchSize, hiddenSize)));
                exe.AssignTensor(lstm.HX, h0.AsTensor(Shape.Create(1, batchSize, hiddenSize)));

                // set weigths, cuDNN matrices order: IFAO
                var w = exe.GetTensor(lstm.W).Reshape(inputSize*4 + hiddenSize*4 + 2*4, hiddenSize);
                SetWeights(ctx, w, inputSize, hiddenSize);

                exe.Forward();

                y1 = exe.GetTensor(lstm.Y).ToArray3D();
                cy1 = exe.GetTensor(lstm.CY).Reshape(batchSize, hiddenSize).ToArray2D();
                hy1 = exe.GetTensor(lstm.HY).Reshape(batchSize, hiddenSize).ToArray2D();

                exe.AssignGradientDirectly(lstm.Y, dy.AsTensor());

                exe.Backward();

                dx1 = exe.GetGradient(lstm.X).ToArray3D();
                dcx1 = exe.GetGradient(lstm.CX).Reshape(batchSize, hiddenSize).ToArray2D();
                dhx1 = exe.GetGradient(lstm.HX).Reshape(batchSize, hiddenSize).ToArray2D();            
                dw1 = exe.GetGradient(lstm.W).ToArray(); // cuDNN weight is 1D linear blob
            }

            {
                var x = Variable<float>(PartialShape.Create(-1, -1, inputSize));
                var lstm = new RnnDynamic<float>(new LstmRnnType(), x, 1, hiddenSize, dropout: 0.0);
                var exe = new Executor(ctx, lstm.Y);
                exe.Initalize();

                // set input
                exe.AssignTensor(lstm.X, data.AsTensor());

                // set states
                exe.AssignTensor(lstm.CX, c0.AsTensor(Shape.Create(1, batchSize, hiddenSize)));
                exe.AssignTensor(lstm.HX, h0.AsTensor(Shape.Create(1, batchSize, hiddenSize)));

                // set weigths, cuDNN matrices order: IFAO
                var w = exe.GetTensor(lstm.W).Reshape(inputSize*4 + hiddenSize*4 + 2*4, hiddenSize);
                SetWeights(ctx, w, inputSize, hiddenSize);

                exe.Forward();

                y2 = exe.GetTensor(lstm.Y).ToArray3D();
                cy2 = exe.GetTensor(lstm.CY).Reshape(batchSize, hiddenSize).ToArray2D();
                hy2 = exe.GetTensor(lstm.HY).Reshape(batchSize, hiddenSize).ToArray2D();

                exe.AssignGradientDirectly(lstm.Y, dy.AsTensor());

                exe.Backward();

                dx2 = exe.GetGradient(lstm.X).ToArray3D();
                dcx2 = exe.GetGradient(lstm.CX).Reshape(batchSize, hiddenSize).ToArray2D();
                dhx2 = exe.GetGradient(lstm.HX).Reshape(batchSize, hiddenSize).ToArray2D();
                dw2 = exe.GetGradient(lstm.W).ToArray();
            }

            AreClose(y1, y2, error);
            AreClose(cy1, cy2, error);
            AreClose(hy1, hy2, error);
            AreClose(dx1, dx2, error);
            AreClose(dcx1, dcx2, error);
            AreClose(dhx1, dhx2, error);
            AreClose(dw1, dw2, error);
        }

        [Test]
        public static void RnnAgainstIteratedRnnCell()
        {
            var ctx = Context.GpuContext(0);
            var inputSize = 5;
            var seqLength = 3;
            var batchSize = 2;
            var hiddenSize = 4;
            var error = 1e-5;

            var data = Context.CpuContext.Eval(RandomUniform<float>(-1, 1, Shape.Create(seqLength, batchSize, inputSize))).ToArray3D();
            data.AsTensor(Shape.Create(seqLength*batchSize, inputSize)).Print();

            var h0 = Context.CpuContext.Eval(RandomNormal<float>(Shape.Create(batchSize, hiddenSize))).ToArray2D();
            var c0 = Context.CpuContext.Eval(RandomNormal<float>(Shape.Create(batchSize, hiddenSize))).ToArray2D();
            var dOUtput = Context.CpuContext.Eval(RandomUniform<float>(-1, 1, Shape.Create(seqLength, batchSize, hiddenSize))).ToArray3D();

            float[,,] y1, y2, dx1, dx2;
            float[,] cy1, cy2, hy1, hy2;
            float[,] dcx1, dcx2, dhx1, dhx2;
            float[] dw1, dw2;

            {
                var x = Variable<float>(PartialShape.Create(seqLength, batchSize, inputSize));
                var lstm = new Rnn<float>(new LstmRnnType(), x, 1, hiddenSize, dropout: 0.0);
                var exe = new Executor(ctx, lstm.Y);
                exe.Initalize();

                // set input
                exe.AssignTensor(lstm.X, data.AsTensor());

                // set states
                exe.AssignTensor(lstm.CX, c0.AsTensor(Shape.Create(1, batchSize, hiddenSize)));
                exe.AssignTensor(lstm.HX, h0.AsTensor(Shape.Create(1, batchSize, hiddenSize)));

                // set weigths, cuDNN matrices order: IFAO
                var w = exe.GetTensor(lstm.W).Reshape(inputSize * 4 + hiddenSize * 4 + 2 * 4, hiddenSize);
                SetWeights(ctx, w, inputSize, hiddenSize);

                exe.Forward();

                y1 = exe.GetTensor(lstm.Y).ToArray3D();
                cy1 = exe.GetTensor(lstm.CY).Reshape(batchSize, hiddenSize).ToArray2D();
                hy1 = exe.GetTensor(lstm.HY).Reshape(batchSize, hiddenSize).ToArray2D();

                exe.AssignGradientDirectly(lstm.Y, dOUtput.AsTensor());

                exe.Backward();

                dx1 = exe.GetGradient(lstm.X).ToArray3D();
                dcx1 = exe.GetGradient(lstm.CX).Reshape(batchSize, hiddenSize).ToArray2D();
                dhx1 = exe.GetGradient(lstm.HX).Reshape(batchSize, hiddenSize).ToArray2D();             
                dw1 = exe.GetGradient(lstm.W).ToArray(); // cuDNN weight is 1D linear blob

            }

            {
                var x = Variable<float>(PartialShape.Create(seqLength, batchSize, inputSize));
                var lstm = new IteratedRnnCell<float>(new LstmRnnType(), x, 1, hiddenSize, true, 0.0);
                var exe = new Executor(ctx, lstm.Output);
                exe.Initalize();

                // set input
                exe.AssignTensor(lstm.Input, data.AsTensor());

                lstm.AssignInitialStates(exe, h0.AsTensor(Shape.Create(1, batchSize, hiddenSize)), c0.AsTensor(Shape.Create(1, batchSize, hiddenSize)));

                // set weigths, cuDNN matrices order: IFAO
                var w = exe.GetTensor(lstm.W).Reshape(inputSize*4 + hiddenSize*4 + 2*4, hiddenSize);
                SetWeights(ctx, w, inputSize, hiddenSize);

                exe.Forward();

                y2 = exe.GetTensor(lstm.Output).ToArray3D();
                cy2 = exe.GetTensor(lstm.CY).Reshape(batchSize, hiddenSize).ToArray2D();
                hy2 = exe.GetTensor(lstm.HY).Reshape(batchSize, hiddenSize).ToArray2D();

                exe.AssignGradientDirectly(lstm.Output, dOUtput.AsTensor());

                exe.Backward();

                dx2 = exe.GetGradient(lstm.Output).ToArray3D();
                dcx2 = exe.GetGradient(lstm.CX).Reshape(batchSize, hiddenSize).ToArray2D();
                dhx2 = exe.GetGradient(lstm.HX).Reshape(batchSize, hiddenSize).ToArray2D();
                dw2 = exe.GetGradient(lstm.W).ToArray();
            }

            AreClose(y1, y2, error);
            AreClose(cy1, cy2, error);
            AreClose(hy1, hy2, error);
            //AreClose(dx1, dx2, error);  // bugbug
            //AreClose(dcx1, dcx2, error); // bugbug
            //AreClose(dhx1, dhx2, error); // bugbug
            //AreClose(dw1, dw2, error); // bugbug 
        }
    }
}
