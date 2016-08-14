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

            public long AttentionDim { get; }
            public long EncoderHiddenSize { get; }
            public long DecoderHiddenSize { get; }
            public long EncoderSeqLength { get; }
            public long Batch { get; }

            public Variable<T> Output { get; }

            public Attention(Variable<T> encoderHiddenStates, Variable<T> decoderHiddenStates, long attentionDim)
            {
                AttentionDim = attentionDim;
                EncoderHiddenStates = encoderHiddenStates;
                DecoderHiddenStates = decoderHiddenStates;

                Util.EnsureEqual(3, EncoderHiddenStates.Shape.Rank, "EncoderHiddenStates layout: (seqLength, batch, encoderHiddenSize)");
                Util.EnsureTrue(EncoderHiddenStates.Shape[0] > 0, "EncoderSeqLength should be determined.");
                //Util.EnsureTrue(EncoderHiddenStates.Shape[1] > 0, "Batch should be determined.");
                Util.EnsureTrue(EncoderHiddenStates.Shape[2] > 0, "EncoderHiddenStates should be determined.");
                EncoderSeqLength = EncoderHiddenStates.Shape[0];
                Batch = EncoderHiddenStates.Shape[1];
                EncoderHiddenSize = EncoderHiddenStates.Shape[2];

                Util.EnsureEqual(2, DecoderHiddenStates.Shape.Rank, "DecoderHiddenStates layout: (batch, decoderHiddenSize)");
                //Util.EnsureTrue(DecoderHiddenStates.Shape[0] > 0, "Batch should be determined.");
                Util.EnsureTrue(DecoderHiddenStates.Shape[1] > 0, "DecoderHiddenStates should be determined.");
                Util.EnsureEqual(Batch, DecoderHiddenStates.Shape[0], "Batch not match.");
                DecoderHiddenSize = DecoderHiddenStates.Shape[1];

                var scaleWh = Sqrt(12.0.AsScalar<T>() / ((double)(AttentionDim + EncoderHiddenSize)).AsScalar<T>());
                Wh = Parameter(scaleWh * (RandomUniform<T>(Shape.Create(EncoderHiddenSize, AttentionDim), 0UL, 0UL) - 0.5.AsScalar<T>()));

                var scaleWd = Sqrt(12.0.AsScalar<T>()/((double) (AttentionDim + DecoderHiddenSize)).AsScalar<T>());
                Wd = Parameter(scaleWd * (RandomUniform<T>(Shape.Create(DecoderHiddenSize, AttentionDim), 0UL, 0UL) - 0.5.AsScalar<T>()));

                // build the graph
                var h = EncoderHiddenStates.Reshape(-1, EncoderHiddenSize);
                var d = DecoderHiddenStates;
                var whh = Dot(h, Wh).Reshape(EncoderSeqLength, -1, AttentionDim);
                var wdd = Dot(d, Wd);

                Output = whh + wdd;
            }
        }

        [Test]
        public static void TestAttention()
        {
            var batch = 4;
            var encoderHiddenSize = 5;
            var decoderHiddenSize = 4;
            var encoderSeqLength = 3;
            var attentionDim = 3;

            // (encoderSeqLength, batch, encoderHiddenSize)
            var encoderHiddenStates = Variable<float>(PartialShape.Create(encoderSeqLength, -1, encoderHiddenSize));
            var decoderHiddenStates = Variable<float>(PartialShape.Create(-1, decoderHiddenSize));
            var attention = new Attention<float>(encoderHiddenStates, decoderHiddenStates, attentionDim);

            var ctx = Context.GpuContext(0);
            var exe = new Executor(ctx, attention.Output);
            exe.Initalize();

            var dataEncoderHiddenStates = new float[encoderSeqLength, batch, encoderHiddenSize];
            RandArray(dataEncoderHiddenStates);

            var dataDecoderHiddenStates = new float[batch, decoderHiddenSize];
            RandArray(dataDecoderHiddenStates);

            exe.AssignTensor(encoderHiddenStates, dataEncoderHiddenStates.AsTensor());
            exe.AssignTensor(decoderHiddenStates, dataDecoderHiddenStates.AsTensor());
            exe.Forward();

            var tensorOutput = exe.GetTensor(attention.Output);
            Console.WriteLine(tensorOutput.Shape);
            tensorOutput.Reshape(encoderSeqLength*batch, -1).Print();

            var dataDOutput = new float[encoderSeqLength, batch, attentionDim];
            RandArray(dataDOutput);
            exe.AssignGradientDirectly(attention.Output, dataDOutput.AsTensor());
            exe.Backward();

            var tensorDWh = exe.GetGradient(attention.Wh);
            tensorDWh.Print();
        }
    }
}
