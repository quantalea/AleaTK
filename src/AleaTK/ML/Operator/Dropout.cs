using System;
using System.Linq;
using Alea;
using static AleaTK.Library;
using static AleaTK.ML.Library;

namespace AleaTK.ML.Operator
{
    public class Dropout<T> : Differentiable
    {
        public Dropout(Variable<T> input, int embedSize, int embedDim, double dropoutRate = 0.5)
        {
            Util.EnsureTrue(dropoutRate > 0.0);
            Util.EnsureTrue(dropoutRate < 1.0);

            Input = input;
            Output = Library.Variable<T>(input.Shape);

            double scale = 1.0 / (1.0 - dropoutRate);

            //Mask = AuxVariable(RandomUniform<int>(input.Shape));

            AddInput(input);
            AddAuxVar(Mask);
            AddOutput(Output);
        }

        public Variable<T> Input { get; }

        public Variable<T> Output { get; }

        public Variable<int> Mask { get; }

        public override void Backward(Executor executor)
        {
            throw new NotImplementedException();
        }

        public override void Forward(Executor executor)
        {
            throw new NotImplementedException();
        }
    }
}