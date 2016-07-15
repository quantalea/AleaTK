using static AleaTK.Library;
using static AleaTK.ML.Library;

namespace AleaTK.ML.Operator
{
    public class L2Loss<T> : Differentiable
    {
        public L2Loss(Variable<T> pred, Variable<T> label)
        {
            Pred = pred;
            Label = label;
            Loss = Variable<T>();
            AddInput(pred);
            AddInput(label);
            AddOutput(Loss);
        }

        public Variable<T> Pred { get; }

        public Variable<T> Label { get; }

        public Variable<T> Loss { get; }

        public override void Forward(Executor executor)
        {
            var pred = executor.GetTensor(Pred);
            var label = executor.GetTensor(Label);
            executor.AssignTensor(Loss, ReduceSum((pred - label) * (pred - label)));
        }

        public override void Backward(Executor executor)
        {
            var pred = executor.GetTensor(Pred);
            var label = executor.GetTensor(Label);
            executor.AssignGradient(Pred, 2.0.AsScalar<T>() * (pred - label));
        }
    }
}