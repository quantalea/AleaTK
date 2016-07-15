using static AleaTK.Library;
using static AleaTK.ML.Library;

namespace AleaTK.ML.Operator
{
    public class ActivationReLU<T> : Differentiable
    {
        public ActivationReLU(Variable<T> input)
        {
            Input = input;
            Output = Variable<T>(input.Shape);
            AddInput(Input);
            AddOutput(Output);
        }

        public Variable<T> Input { get; }

        public Variable<T> Output { get; }

        public override void Forward(Executor executor)
        {
            var input = executor.GetTensor(Input);
            executor.AssignTensor(Output, Max(input, 0.0.AsScalar<T>()));
        }

        public override void Backward(Executor executor)
        {
            var output = executor.GetTensor(Output);
            var dOutput = executor.GetGradient(Output);
            executor.AssignGradient(Input, ReLUGrad(output) * dOutput);
        }
    }

    public class ActivationSigmoid<T> : Differentiable
    {
        public ActivationSigmoid(Variable<T> input)
        {
            Input = input;
            Output = Variable<T>(input.Shape);
            AddInput(Input);
            AddOutput(Output);
        }

        public Variable<T> Input { get; }

        public Variable<T> Output { get; }

        public override void Forward(Executor executor)
        {
            var input = executor.GetTensor(Input);
            executor.AssignTensor(Output, 1.0.AsScalar<T>() / (1.0.AsScalar<T>() + Exp(-input)));
        }

        public override void Backward(Executor executor)
        {
            var output = executor.GetTensor(Output);
            var dOutput = executor.GetGradient(Output);
            executor.AssignGradient(Input, output * (1.0.AsScalar<T>() - output) * dOutput);
        }
    }

    public class ActivationTanh<T> : Differentiable
    {
        public ActivationTanh(Variable<T> input)
        {
            Input = input;
            Output = Variable<T>(input.Shape);
            AddInput(Input);
            AddOutput(Output);
        }

        public Variable<T> Input { get; }

        public Variable<T> Output { get; }

        public override void Forward(Executor executor)
        {
            var input = executor.GetTensor(Input);
            executor.AssignTensor(Output, Tanh(input));
        }

        public override void Backward(Executor executor)
        {
            var output = executor.GetTensor(Output);
            var dOutput = executor.GetGradient(Output);
            executor.AssignGradient(Input, (1.0.AsScalar<T>() - output * output) * dOutput);
        }
    }
}
