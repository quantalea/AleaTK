using System.Linq;
using static AleaTK.Library;
using static AleaTK.ML.Library;

namespace AleaTK.ML.Operator
{
    public class Add<T> : Differentiable
    {
        public Add(Variable<T> a, Variable<T> b)
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
            executor.AssignTensor(C, a + b);
        }

        public override void Backward(Executor executor)
        {
            var a = executor.GetTensor(A);
            var b = executor.GetTensor(B);
            var dC = executor.GetGradient(C);

            var dA = a.Shape.Rank < dC.Shape.Rank
                ? ReduceSum(dC, Enumerable.Range(0, dC.Shape.Rank - a.Shape.Rank).ToArray())
                : dC;

            var dB = b.Shape.Rank < dC.Shape.Rank
                ? ReduceSum(dC, Enumerable.Range(0, dC.Shape.Rank - b.Shape.Rank).ToArray())
                : dC;
            
            executor.AssignGradient(A, dA);
            executor.AssignGradient(B, dB);
        }
    }
}
