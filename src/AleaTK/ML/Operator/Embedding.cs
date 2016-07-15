using System.Linq;
using static AleaTK.Library;

namespace AleaTK.ML.Operator
{
    public class Embedding<T> : Differentiable
    {
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
            var indices = executor.GetTensor(Indices);
            var gradout = executor.GetGradient(Output);
            executor.AssignGradient(Weights, TakeGrad(indices, gradout, EmbedSize));
        }
    }
}