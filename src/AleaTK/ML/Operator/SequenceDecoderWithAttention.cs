using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Alea;
using Alea.cuDNN;
using Alea.Parallel;
using static AleaTK.Library;
using static AleaTK.ML.Library;

namespace AleaTK.ML.Operator
{
    public class SequenceDecoderWithAttention<T> : Differentiable
    {
        public SequenceDecoderWithAttention(int encoderOutputSize)
        {
            // Y Shape (maxSeqLength, not yet known, hiddenSize)
            EncoderOutput = Variable<T>(PartialShape.Create(-1, -1, encoderOutputSize));
        }

        public override void Forward(Executor executor)
        {
            throw new NotImplementedException();
        }

        public override void Backward(Executor executor)
        {
            throw new NotImplementedException();
        }

        public Variable<T> EncoderOutput { get; }
    }
}
