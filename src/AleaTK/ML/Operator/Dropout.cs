using System;
using System.Linq;
using Alea;
using static AleaTK.Library;
using static AleaTK.ML.Library;

namespace AleaTK.ML.Operator
{
    /// <summary>
    /// With probability (1 - dropoutProb) outputs the input element scaled up by 1 / (1 - dropoutProb), 
    /// otherwise outputs 0. The scaling is so that the expected sum is unchanged.
    /// </summary>
    public class Dropout<T> : Differentiable
    {
        public Dropout(Variable<T> input, double dropoutProb = 0.5)
        {
            Util.EnsureTrue(dropoutProb > 0.0);
            Util.EnsureTrue(dropoutProb < 1.0);

            Input = input;
            Output = Library.Variable<T>(input.Shape);

            Scale = 1.0 / (1.0 - dropoutProb);

            Threshold = (uint) ((double) UInt32.MaxValue*dropoutProb);

            Mask = AuxVariable<uint>();

            AddInput(input);
            AddAuxVar(Mask);
            AddOutput(Output);
        }

        public Variable<T> Input { get; }

        public Variable<T> Output { get; }

        public Variable<uint> Mask { get; }

        public uint Threshold { get; }

        public double Scale { get; }

        public override void Forward(Executor executor)
        {
            var ctx = executor.Context;
            var input = executor.GetTensor(Input);
            // TODO: make sure the offset is correct in one training.
            executor.AssignTensor(Mask, RandomUniform<uint>(input.Shape));
            var mask = executor.GetTensor(Mask);
            executor.AssignTensor(Output, Dropout(input, mask, Threshold, Scale));
        }

        public override void Backward(Executor executor)
        {
            var dOutput = executor.GetGradient(Output);
            var mask = executor.GetTensor(Mask);
            executor.AssignGradient(Input, Dropout(dOutput, mask, Threshold, Scale));
        }
    }
}