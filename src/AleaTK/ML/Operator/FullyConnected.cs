using System;
using System.Linq;
using static AleaTK.Library;
using static AleaTK.ML.Library;

namespace AleaTK.ML.Operator
{
    public class FullyConnected<T> : Differentiable
    {
        public FullyConnected(Variable<T> data, long numHidden)
        {
            Util.EnsureTrue(data.HasShape);
            Util.EnsureEqual(2, data.Shape.Rank, "Input must be matrix.");
            Util.EnsureTrue(data.Shape[1] > 0L);

            Data = data;

            var numInput = data.Shape[1];
            var scale = Sqrt(12.0.AsScalar<T>() / ((double)(numInput + numHidden)).AsScalar<T>());
            Weights = Parameter(scale * (RandomUniform<T>(Shape.Create(numInput, numHidden), 0UL, 0UL) - 0.5.AsScalar<T>()));

            Bias = Parameter(Fill(Shape.Create(numHidden), ScalarOps.Conv<T>(0.0)));
            Output = Variable<T>(PartialShape.Create(data.Shape[0], numHidden));

            AddInput(Data);
            AddInput(Weights);
            AddInput(Bias);
            AddOutput(Output);
        }

        public Variable<T> Data { get; }

        public Variable<T> Weights { get; }

        public Variable<T> Bias { get; }

        public Variable<T> Output { get; }

        public override void Forward(Executor executor)
        {
            var data = executor.GetTensor(Data);
            var weights = executor.GetTensor(Weights);
            var bias = executor.GetTensor(Bias);
            executor.AssignTensor(Output, Dot(data.Reshape(data.Shape[0], -1), weights) + bias);
        }

        public override void Backward(Executor executor)
        {
            var data = executor.GetTensor(Data);
            var weights = executor.GetTensor(Weights);
            var dOutput = executor.GetGradient(Output);
            executor.AssignGradient(Data, Dot(dOutput, weights.T).Reshape(data.Shape.AsArray));
            executor.AssignGradient(Weights, Dot(data.Reshape(data.Shape[0], -1).T, dOutput));
            executor.AssignGradient(Bias, ReduceSum(dOutput, 0));
        }
    }
}
