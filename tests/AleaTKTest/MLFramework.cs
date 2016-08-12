using System;
using System.CodeDom.Compiler;
using System.Collections.Generic;
using System.Data;
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
    }
}
