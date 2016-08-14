using System;
using System.CodeDom.Compiler;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AleaTK;
using AleaTK.ML;
using AleaTK.ML.Operator;
using NUnit.Framework;
using static AleaTK.Library;
using static AleaTK.ML.Library;
using Executor = AleaTK.ML.Executor;

namespace AleaTKTest
{
    public static class MLFramework
    {
        public class ImperativeLoopOp : Differentiable
        {
            public int SeqLength { get; }
            public long Size { get; }

            public Variable<float> Input { get; }
            public Variable<float> Weight { get; }
            public Variable<float> Output { get; }
            public Variable<float> Temp { get; }

            public Variable<float> SubInput { get; }
            public Variable<float> SubWeight { get; }
            public Variable<float> SubOutput { get; }

            public Symbol SubExecutor { get; } = new Symbol();

            public ImperativeLoopOp(Variable<float> input, int seqLength)
            {
                SeqLength = seqLength;

                Input = input;
                var size = input.Shape[1];
                Size = size;

                Weight = Parameter<float>(Fill(Shape.Create(seqLength, size, size), 1.0f));

                Output = Variable<float>(PartialShape.Create(size, size));

                Temp = AuxVariable<float>();

                AddInput(Input);
                AddInput(Weight);
                AddOutput(Output);
                AddAuxVar(Temp);

                SubInput = Variable<float>(input.Shape);
                SubWeight = Variable<float>(input.Shape);
                SubOutput = Dot(SubInput, SubWeight);
            }

            public override void Initialize(Executor executor)
            {
                var subExecutor = new Executor(executor.Context, SubOutput);
                executor.Objects[SubExecutor] = subExecutor;
                subExecutor.Initalize();
                base.Initialize(executor);
            }

            public override void Forward(Executor executor)
            {
                var subExecutor = (Executor)executor.Objects[SubExecutor];
                var input = executor.GetTensor(Input);
                var output = executor.GetTensor(Output, Shape.Create(Size, Size));
                var temp = executor.GetTensor(Temp, Shape.Create(SeqLength - 1, Size, Size));

                for (var t = 0; t < SeqLength; ++t)
                {
                    
                }


                throw new NotImplementedException();
            }

            public override void Backward(Executor executor)
            {
                throw new NotImplementedException();
            }
        }

        [Test]
        public static void TestImperativeLoopOp()
        {
            var ctx = Context.GpuContext(0);

            var size = 100;
            var input = Variable<float>(PartialShape.Create(size, size));
            var loop = new ImperativeLoopOp(input, size);

            var exe = new Executor(ctx, loop.Output);
            exe.Initalize();

            var inputHost = new float[size, size];
            var rng = new Random();
            for (var i = 0; i < size; ++i)
            {
                for (var j = 0; j < size; ++j)
                {
                    inputHost[i, j] = (float)rng.NextDouble();
                }
            }

            exe.AssignTensor(loop.Input, inputHost.AsTensor());
            exe.Forward();


        }

        private static readonly Random _rng = new Random();

        private static void RandArray(float[,] array)
        {
            for (var i = 0; i < array.GetLength(0); ++i)
            {
                for (var j = 0; j < array.GetLength(1); ++j)
                {
                    array[i, j] = (float)_rng.NextDouble();
                }
            }
        }

        private static void RandArray(float[,,] array)
        {
            for (var i = 0; i < array.GetLength(0); ++i)
            {
                for (var j = 0; j < array.GetLength(1); ++j)
                {
                    for (var k = 0; k < array.GetLength(2); ++k)
                    {
                        array[i, j, k] = (float)_rng.NextDouble();
                    }
                }
            }
        }

        public class Attention<T>
        {
            public Variable<T> EncoderHiddenStates { get; }
            public Variable<T> DecoderHiddenStates { get; }

            public Variable<T> Wh { get; }
            public Variable<T> Wd { get; }
            public Variable<T> V { get; }

            public long AttentionDim { get; }
            public long EncoderHiddenSize { get; }
            public long DecoderHiddenSize { get; }
            public long Batch { get; }

            public Variable<T> Output { get; }

            public Attention(Variable<T> encoderHiddenStates, Variable<T> decoderHiddenStates, long attentionDim)
            {
                AttentionDim = attentionDim;
                EncoderHiddenStates = encoderHiddenStates;
                DecoderHiddenStates = decoderHiddenStates;

                // one goal is, try to make batchSize and encoderSeqLength unknown at symbol layer
                // so, in LSTM outer op, we can create one graph and one sub-executor, and applied for
                // different encoderSeqLength and batchSize.
                Util.EnsureEqual(3, EncoderHiddenStates.Shape.Rank, "EncoderHiddenStates layout: (encoderSeqLength, batch, encoderHiddenSize)");
                Util.EnsureTrue(EncoderHiddenStates.Shape[2] > 0, "EncoderHiddenStates should be determined.");
                EncoderHiddenSize = EncoderHiddenStates.Shape[2];

                Util.EnsureEqual(2, DecoderHiddenStates.Shape.Rank, "DecoderHiddenStates layout: (batch, decoderHiddenSize)");
                Util.EnsureTrue(DecoderHiddenStates.Shape[1] > 0, "DecoderHiddenStates should be determined.");
                DecoderHiddenSize = DecoderHiddenStates.Shape[1];

                var scaleWh = Sqrt(12.0.AsScalar<T>() / ((double)(AttentionDim + EncoderHiddenSize)).AsScalar<T>());
                Wh = Parameter(scaleWh * (RandomUniform<T>(Shape.Create(EncoderHiddenSize, AttentionDim), 0UL, 0UL) - 0.5.AsScalar<T>()));

                var scaleWd = Sqrt(12.0.AsScalar<T>() / ((double)(AttentionDim + DecoderHiddenSize)).AsScalar<T>());
                Wd = Parameter(scaleWd * (RandomUniform<T>(Shape.Create(DecoderHiddenSize, AttentionDim), 0UL, 0UL) - 0.5.AsScalar<T>()));

                var scaleV = Sqrt(12.0.AsScalar<T>() / ((double)(AttentionDim)).AsScalar<T>());
                V = Parameter(scaleV * (RandomUniform<T>(Shape.Create(AttentionDim, 1), 0UL, 0UL) - 0.5.AsScalar<T>()));

                // build the graph
                var h = EncoderHiddenStates.Reshape(-1, EncoderHiddenSize); // (n*b,He) // He denotes hiddenSize of encoder
                var d = DecoderHiddenStates; // (b,Hd) // Hd denotes hiddenSize of decoder
                var whh = Dot(h, Wh); // shape (n*b,K) K denotes attentionDim
                var wdd = Dot(d, Wd); // shape (b,K)

                // to add whh and wdd, we need broadcast, for this, we need to know at least n or b.
                // The decision here is to make b known at symbolic layer, because then you can have
                // flexibility on n (EncoderSeqLength), easier for making bucket.
                // another issue is, our backward of add has some issue dealing with 3d array which has broadcast
                // so, we can reshape them into 2d tensor here:
                // initial shape: (n*b,K) + (b,K)
                // reshape for the boadcast: (n,b*K) + (b*K) (for broadcasting, (b*K) will broadcast to (1,b*K)
                // then: (n,b*K) + (b*K) = (n,b*K)
                // reshape result to (n*b,K)
                Batch = EncoderHiddenStates.Shape[1];
                Util.EnsureTrue(Batch > 0, "Batch need to be determined.");
                Util.EnsureTrue(Batch == DecoderHiddenStates.Shape[0]);
                var add = (whh.Reshape(-1, Batch * AttentionDim) + wdd.Reshape(-1)).Reshape(-1, AttentionDim);

                // tanh, shape no change (n*b,K)
                var whd = new ActivationTanh<T>(add);

                // (n*b,K) dot (K,1) = (n*b,1) => reshape to (n,b)
                var u = Dot(whd.Output, V).Reshape(-1, Batch);

                // same shape (n,b)
                var softmax = new Softmax<T>(u);

                Output = softmax.Output;
            }
        }

        [Test]
        public static void TestAttention()
        {
            var batch = 4;
            var encoderHiddenSize = 5;
            var decoderHiddenSize = 4;
            var attentionDim = 3;

            // (encoderSeqLength, batch, encoderHiddenSize)
            var encoderHiddenStates = Variable<float>(PartialShape.Create(-1, batch, encoderHiddenSize));
            var decoderHiddenStates = Variable<float>(PartialShape.Create(batch, decoderHiddenSize));
            var attention = new Attention<float>(encoderHiddenStates, decoderHiddenStates, attentionDim);

            var ctx = Context.GpuContext(0);
            var exe = new Executor(ctx, attention.Output) { AssignAllGradient = true };
            exe.Initalize();

            // encoderSeqLength is flexibly at runtime
            var encoderSeqLength = 3;
            var dataEncoderHiddenStates = new float[encoderSeqLength, batch, encoderHiddenSize];
            RandArray(dataEncoderHiddenStates);

            var dataDecoderHiddenStates = new float[batch, decoderHiddenSize];
            RandArray(dataDecoderHiddenStates);

            exe.AssignTensor(encoderHiddenStates, dataEncoderHiddenStates.AsTensor());
            exe.AssignTensor(decoderHiddenStates, dataDecoderHiddenStates.AsTensor());
            exe.Forward();

            var tensorOutput = exe.GetTensor(attention.Output);
            Console.WriteLine(tensorOutput.Shape);
            tensorOutput.Reshape(encoderSeqLength * batch, -1).Print();

            var dataDOutput = new float[encoderSeqLength, batch];
            RandArray(dataDOutput);
            exe.AssignGradientDirectly(attention.Output, dataDOutput.AsTensor());
            exe.Backward();

            var tensorDWh = exe.GetGradient(attention.Wh);
            tensorDWh.Print();

            var tensorDWd = exe.GetGradient(attention.Wd);
            tensorDWd.Print();

            var tensorDH = exe.GetGradient(attention.EncoderHiddenStates);
            Console.WriteLine(tensorDH.Shape);
            tensorDH.Reshape(-1, encoderHiddenSize).Print();

            var tensorDD = exe.GetGradient(attention.DecoderHiddenStates);
            Console.WriteLine(tensorDD.Shape);
            tensorDD.Print();
        }

        public class AttentionReduce<T> : Differentiable
        {
            public AttentionReduce(Variable<T> softmax, Variable<T> encoderHiddenStates)
            {
                Softmax = softmax;
                EncoderHiddenStates = encoderHiddenStates;
                Batch = EncoderHiddenStates.Shape[1];
                EncoderHiddenSize = EncoderHiddenStates.Shape[2];
                Output = Variable<T>(PartialShape.Create(Batch, EncoderHiddenSize));

                AddInput(Softmax);
                AddInput(EncoderHiddenStates);
                AddOutput(Output);
            }

            public Variable<T> Softmax { get; }

            public Variable<T> EncoderHiddenStates { get; }

            public Variable<T> Output { get; }

            public long Batch { get; }

            public long EncoderHiddenSize { get; }

            public override void Forward(Executor executor)
            {
                var h = executor.GetTensor(EncoderHiddenStates);
                var a = executor.GetTensor(Softmax);
                var n = h.Shape[0];
                var b = Batch;
                var d = EncoderHiddenSize;
                //var y = executor.GetTensor(Output, Shape.Create(b, d));

                var mul = (a.Reshape(n*b, 1)*h.Reshape(n*b, d)).Reshape(n, b*d);
                var reduce = ReduceSum(mul, 0).Reshape(b, d);

                executor.AssignTensor(Output, reduce);
            }

            public override void Backward(Executor executor)
            {
                throw new NotImplementedException();
            }
        }

        public class Attention2<T>
        {
            public Variable<T> EncoderHiddenStates { get; }
            public Variable<T> DecoderHiddenStates { get; }

            public Variable<T> Wh { get; }
            public Variable<T> Wd { get; }
            public Variable<T> V { get; }

            public long AttentionDim { get; }
            public long EncoderHiddenSize { get; }
            public long DecoderHiddenSize { get; }
            public long Batch { get; }

            public Variable<T> Output { get; }

            public Attention2(Variable<T> encoderHiddenStates, Variable<T> decoderHiddenStates, long attentionDim)
            {
                AttentionDim = attentionDim;
                EncoderHiddenStates = encoderHiddenStates;
                DecoderHiddenStates = decoderHiddenStates;

                // one goal is, try to make batchSize and encoderSeqLength unknown at symbol layer
                // so, in LSTM outer op, we can create one graph and one sub-executor, and applied for
                // different encoderSeqLength and batchSize.
                Util.EnsureEqual(3, EncoderHiddenStates.Shape.Rank, "EncoderHiddenStates layout: (encoderSeqLength, batch, encoderHiddenSize)");
                Util.EnsureTrue(EncoderHiddenStates.Shape[2] > 0, "EncoderHiddenStates should be determined.");
                EncoderHiddenSize = EncoderHiddenStates.Shape[2];

                Util.EnsureEqual(2, DecoderHiddenStates.Shape.Rank, "DecoderHiddenStates layout: (batch, decoderHiddenSize)");
                Util.EnsureTrue(DecoderHiddenStates.Shape[1] > 0, "DecoderHiddenStates should be determined.");
                DecoderHiddenSize = DecoderHiddenStates.Shape[1];

                var scaleWh = Sqrt(12.0.AsScalar<T>() / ((double)(AttentionDim + EncoderHiddenSize)).AsScalar<T>());
                Wh = Parameter(scaleWh * (RandomUniform<T>(Shape.Create(EncoderHiddenSize, AttentionDim), 0UL, 0UL) - 0.5.AsScalar<T>()));

                var scaleWd = Sqrt(12.0.AsScalar<T>() / ((double)(AttentionDim + DecoderHiddenSize)).AsScalar<T>());
                Wd = Parameter(scaleWd * (RandomUniform<T>(Shape.Create(DecoderHiddenSize, AttentionDim), 0UL, 0UL) - 0.5.AsScalar<T>()));

                var scaleV = Sqrt(12.0.AsScalar<T>() / ((double)(AttentionDim)).AsScalar<T>());
                V = Parameter(scaleV * (RandomUniform<T>(Shape.Create(AttentionDim, 1), 0UL, 0UL) - 0.5.AsScalar<T>()));

                // build the graph
                var h = EncoderHiddenStates.Reshape(-1, EncoderHiddenSize); // (n*b,He) // He denotes hiddenSize of encoder
                var d = DecoderHiddenStates; // (b,Hd) // Hd denotes hiddenSize of decoder
                var whh = Dot(h, Wh); // shape (n*b,K) K denotes attentionDim
                var wdd = Dot(d, Wd); // shape (b,K)

                // to add whh and wdd, we need broadcast, for this, we need to know at least n or b.
                // The decision here is to make b known at symbolic layer, because then you can have
                // flexibility on n (EncoderSeqLength), easier for making bucket.
                // another issue is, our backward of add has some issue dealing with 3d array which has broadcast
                // so, we can reshape them into 2d tensor here:
                // initial shape: (n*b,K) + (b,K)
                // reshape for the boadcast: (n,b*K) + (b*K) (for broadcasting, (b*K) will broadcast to (1,b*K)
                // then: (n,b*K) + (b*K) = (n,b*K)
                // reshape result to (n*b,K)
                Batch = EncoderHiddenStates.Shape[1];
                Util.EnsureTrue(Batch > 0, "Batch need to be determined.");
                Util.EnsureTrue(Batch == DecoderHiddenStates.Shape[0]);
                var add = (whh.Reshape(-1, Batch * AttentionDim) + wdd.Reshape(-1)).Reshape(-1, AttentionDim);

                // tanh, shape no change (n*b,K)
                var whd = new ActivationTanh<T>(add);

                // (n*b,K) dot (K,1) = (n*b,1) => reshape to (n,b)
                var u = Dot(whd.Output, V).Reshape(-1, Batch);

                // same shape (n,b)
                var softmax = new Softmax<T>(u);

                var reduce = new AttentionReduce<T>(softmax.Output, EncoderHiddenStates);

                Output = reduce.Output;
            }
        }

        [Test]
        public static void TestAttention2()
        {
            var batch = 4;
            var encoderHiddenSize = 5;
            var decoderHiddenSize = 4;
            var attentionDim = 3;

            // (encoderSeqLength, batch, encoderHiddenSize)
            var encoderHiddenStates = Variable<float>(PartialShape.Create(-1, batch, encoderHiddenSize));
            var decoderHiddenStates = Variable<float>(PartialShape.Create(batch, decoderHiddenSize));
            var attention = new Attention2<float>(encoderHiddenStates, decoderHiddenStates, attentionDim);

            var ctx = Context.GpuContext(0);
            var exe = new Executor(ctx, attention.Output) { AssignAllGradient = true };
            exe.Initalize();

            // encoderSeqLength is flexibly at runtime
            var encoderSeqLength = 3;
            var dataEncoderHiddenStates = new float[encoderSeqLength, batch, encoderHiddenSize];
            RandArray(dataEncoderHiddenStates);

            var dataDecoderHiddenStates = new float[batch, decoderHiddenSize];
            RandArray(dataDecoderHiddenStates);

            exe.AssignTensor(encoderHiddenStates, dataEncoderHiddenStates.AsTensor());
            exe.AssignTensor(decoderHiddenStates, dataDecoderHiddenStates.AsTensor());
            exe.Forward();

            var tensorOutput = exe.GetTensor(attention.Output);
            Console.WriteLine(tensorOutput.Shape);
            tensorOutput.Print();

            //var dataDOutput = new float[encoderSeqLength, batch];
            //RandArray(dataDOutput);
            //exe.AssignGradientDirectly(attention.Output, dataDOutput.AsTensor());
            //exe.Backward();

            //var tensorDWh = exe.GetGradient(attention.Wh);
            //tensorDWh.Print();

            //var tensorDWd = exe.GetGradient(attention.Wd);
            //tensorDWd.Print();

            //var tensorDH = exe.GetGradient(attention.EncoderHiddenStates);
            //Console.WriteLine(tensorDH.Shape);
            //tensorDH.Reshape(-1, encoderHiddenSize).Print();

            //var tensorDD = exe.GetGradient(attention.DecoderHiddenStates);
            //Console.WriteLine(tensorDD.Shape);
            //tensorDD.Print();
        }
    }
}
