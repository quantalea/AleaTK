using System;
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

namespace AleaTKTest
{
    public static class MLFramework
    {
        public class ImperativeLoopOp : Differentiable
        {
            public Variable<float> Input { get; }
            public Variable<float> Weight { get; }
            public Variable<float> Output { get; }

            public ImperativeLoopOp(Variable<float> input)
            {
                Input = input;
                var size = input.Shape[1];

                Weight = Parameter<float>(Fill(Shape.Create(size, size), 1.0f));

                Output = Variable<float>(input.Shape);

                AddInput(Input);
                AddInput(Weight);
                AddOutput(Output);
            }

            public override void Forward(Executor executor)
            {
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
            var loop = new ImperativeLoopOp(input);

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
