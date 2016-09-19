using System;
using System.Linq;
using Alea;
using Alea.CSharp;
using static AleaTK.Library;

namespace AleaTK.ML.Operator
{
    public class Embedding<T> : Differentiable, ILayer<T> {
        public Embedding(Variable<int> indices, int embedSize, int embedDim, double initScale = 0.5)
        {
            Indices = indices;
            Weights = Library.Parameter((initScale * 2.0).AsScalar<T>()*RandomUniform<T>(Shape.Create(embedSize, embedDim)) - initScale.AsScalar<T>());
            Output = Library.Variable<T>(PartialShape.Create(Indices.Shape.Concat(new long[] { embedDim }).ToArray()));
            EmbedSize = embedSize;
            EmbedDim = embedDim;

            AddInput(Indices);
            AddInput(Weights);
            AddOutput(Output);
        }

        public Variable<int> Indices { get; }

        public Variable<T> Weights { get; }

        public Variable<T> Output { get; }

        public int EmbedSize { get; }

        public int EmbedDim { get; }

        public override void Forward(Executor executor)
        {
            var indices = executor.GetTensor(Indices);
            var weights = executor.GetTensor(Weights);
            executor.AssignTensor(Output, Take(indices, weights));
        }

        public override void Backward(Executor executor)
        {
            var ctx = executor.Context;
            var indices = executor.GetTensor(Indices);
            var gradout = executor.GetGradient(Output);

            // for performance fix.
            if (ctx.Type == ContextType.Gpu && gradout.Layout.IsInnerChangeMostFullyPacked && indices.Layout.IsInnerChangeMostFullyPacked)
            {
                var embedDim = EmbedDim;
                var batchSize = (int)indices.Shape.Length;
                var threadSize = 256;

                // first set all to 0
                executor.AssignGradient(Weights, Fill(executor.GetTensor(Weights).Shape, ScalarOps.Conv<T>(0.0)));
                var dW = executor.GetGradient(Weights);

                // then use a 1 block kernel to update it, cause usually the batch size is not huge, but the embedsize is huge!
                var stream = ctx.ToGpuContext().Stream;
                var iPtr = indices.Buffer.Ptr;

                // the following kernel is for 1 block, so there is no need for synchornization,
                // there could be further optimized.

                if (typeof(T) == typeof(float))
                {
                    var dOPtr = gradout.Buffer.Ptr.Reinterpret<float>();
                    var dWPtr = dW.Buffer.Ptr.Reinterpret<float>();
                    var lp = new LaunchParam(1, threadSize);
                    //Console.WriteLine($"{indices.Shape} {gradout.Shape} {dW.Shape}");
                    stream.Launch(() =>
                    {
                        for (var i = 0; i < batchSize; ++i)
                        {
                            var row = iPtr[i];

                            for (var k = threadIdx.x; k < embedDim; k += blockDim.x)
                            {
                                dWPtr[row * embedDim + k] += dOPtr[i * embedDim + k];
                            }
                        }
                    }, lp);

                    return;
                }

                throw new NotImplementedException();
            }
            else
            {
                executor.AssignGradient(Weights, TakeGrad(indices, gradout, EmbedSize));
            }
        }
    }
}