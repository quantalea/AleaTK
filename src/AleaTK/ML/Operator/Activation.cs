using static AleaTK.Library;
using static AleaTK.ML.Library;

namespace AleaTK.ML.Operator
{
    public abstract class Activation<T> : Differentiable
    {
        protected Activation(Variable<T> input)
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
            executor.AssignTensor(Output, ForwardExpr(input));
        }

        protected abstract Expr<T> ForwardExpr(Tensor<T> input);

        public override void Backward(Executor executor)
        {
            var output = executor.GetTensor(Output);
            var dOutput = executor.GetGradient(Output);
            executor.AssignGradient(Input, BackwardExpr(output)*dOutput);
        }

        protected abstract Expr<T> BackwardExpr(Tensor<T> output);
    }

    public class ActivationReLU<T> : Activation<T>
    {
        public ActivationReLU(Variable<T> input) : base(input) { }

        protected override Expr<T> ForwardExpr(Tensor<T> input) { return Max(input, 0.0.AsScalar<T>()); }

        protected override Expr<T> BackwardExpr(Tensor<T> output) { return ReLUGrad(output); }
    }

    public class ActivationSigmoid<T> : Activation<T>
    {
        public ActivationSigmoid(Variable<T> input) : base(input) { }

        protected override Expr<T> ForwardExpr(Tensor<T> input)
        {
            return 1.0.AsScalar<T>()/(1.0.AsScalar<T>() + Exp(-input));
        }

        protected override Expr<T> BackwardExpr(Tensor<T> output) { return output*(1.0.AsScalar<T>() - output); }
    }

    public class ActivationTanh<T> : Activation<T>
    {
        public ActivationTanh(Variable<T> input) : base(input) { }

        protected override Expr<T> ForwardExpr(Tensor<T> input) { return Tanh(input); }

        protected override Expr<T> BackwardExpr(Tensor<T> output) { return 1.0.AsScalar<T>() - output*output; }
    }
}
