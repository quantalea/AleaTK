using static AleaTK.Library;
using static AleaTK.ML.Library;

namespace AleaTK.ML.Operator
{
    public class Dot<T> : Differentiable
    {
        public Dot(Variable<T> a, Variable<T> b)
        {
            A = a;
            B = b;
            C = Variable<T>();
            AddInput(a);
            AddInput(b);
            AddOutput(C);
        }

        public Variable<T> A { get; }

        public Variable<T> B { get; }

        public Variable<T> C { get; }

        public override void Forward(Executor executor)
        {
            var a = executor.GetTensor(A);
            var b = executor.GetTensor(B);
            executor.AssignTensor(C, Dot(a, b));
        }

        public override void Backward(Executor executor)
        {
            var a = executor.GetTensor(A);
            var b = executor.GetTensor(B);
            var dC = executor.GetGradient(C);
            executor.AssignGradient(A, Dot(dC, b.T));
            executor.AssignGradient(B, Dot(a.T, dC));
        }
    }
}