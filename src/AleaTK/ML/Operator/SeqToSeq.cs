using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Alea;
using Alea.cuDNN;
using Alea.CSharp;
using Alea.Parallel;
using static AleaTK.Library;
using static AleaTK.ML.Library;

namespace AleaTK.ML.Operator
{
    /// <summary>
    /// Attention mechanism following http://arxiv.org/abs/1412.7449 section 2.1.
    /// 
    /// For h_1, \ldots, h_{T_A} hidden encoder states and $d_t$ hidden encoder state it calculates the context as 
    /// the weighted sum of hidden encoder states:
    /// 
    ///     u_{t i} = v^T tanh(Wh h_i + Wd d_t), u_t \in \mathbb{R}^{T_A}
    ///     a_t = softmax(u_t)
    ///     c_t = sum_{i = 1}^{T_A} a_{t i} h_i, 
    /// 
    /// We could feed the concatenated vector [d_t, c_t] as new hidden state to update the rnn cell, 
    /// or we could use c_t and concatenate it with the input and feed it as input to the rnn cell.
    /// 
    /// EncoderHiddenStates tensor of dimension [seqLength, batch, encoderHiddenSize]
    /// DecoderHiddenState tensor of dimension [batch, decoderHiddenSize]
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class Attention<T> : Differentiable
    {
        public Variable<T> EncoderHiddenStates { get; }
        public Variable<T> DecoderHiddenState { get; }
        public Variable<T> Wh { get; }
        public Variable<T> Wd { get; }
        public Variable<T> V { get; }
        public Variable<T> Softmax { get; }
        public Variable<T> AttentionState { get; }

        public int SeqLength { get; }
        public int Batch { get; }
        public int EncoderHiddenSize { get; }
        public int DecoderHiddenSize { get; }
        public int AttentionDim { get; }

        public Attention(Variable<T> encoderHiddenStates, Variable<T> decoderHiddenState, int attentionDim)
        {
            AttentionDim = attentionDim;
            EncoderHiddenStates = encoderHiddenStates;
            DecoderHiddenState = decoderHiddenState;

            Util.EnsureEqual(3, EncoderHiddenStates.Shape.Rank, "Input layout: (seqLength, batch, encoderHiddenSize)");
            Util.EnsureTrue(EncoderHiddenStates.Shape[0] >= 0, "Input layout: (seqLength, batch, encoderHiddenSize)");
            Util.EnsureTrue(EncoderHiddenStates.Shape[1] >= 0, "Input layout: (seqLength, batch, encoderHiddenSize)");
            Util.EnsureTrue(EncoderHiddenStates.Shape[2] >= 0, "Input layout: (seqLength, batch, encoderHiddenSize)");
            SeqLength = (int) EncoderHiddenStates.Shape[0];
            Batch = (int) EncoderHiddenStates.Shape[1];
            EncoderHiddenSize = (int) EncoderHiddenStates.Shape[2];

            Util.EnsureEqual(2, DecoderHiddenState.Shape.Rank, "Input layout: (batch, decoderHiddenSize)");
            Util.EnsureTrue(DecoderHiddenState.Shape[0] >= 0, "Input layout: (seqLength, batch, encoderHiddenSize)");
            Util.EnsureTrue(DecoderHiddenState.Shape[1] >= 0, "Input layout: (seqLength, batch, encoderHiddenSize)");
            Util.EnsureTrue(DecoderHiddenState.Shape[0] == EncoderHiddenStates.Shape[0]);
            DecoderHiddenSize = (int) DecoderHiddenState.Shape[1];

            var scale = Sqrt(12.0.AsScalar<T>() / ((double)(AttentionDim + EncoderHiddenSize)).AsScalar<T>());
            Wh = Parameter(scale * (RandomUniform<T>(Shape.Create(EncoderHiddenSize, AttentionDim), 0UL, 0UL) - 0.5.AsScalar<T>()));

            scale = Sqrt(12.0.AsScalar<T>() / ((double)(AttentionDim + DecoderHiddenSize)).AsScalar<T>());
            Wd = Parameter(scale * (RandomUniform<T>(Shape.Create(DecoderHiddenSize, AttentionDim), 0UL, 0UL) - 0.5.AsScalar<T>()));

            scale = Sqrt(12.0.AsScalar<T>() / ((double)(AttentionDim)).AsScalar<T>());
            V = Parameter(scale * (RandomUniform<T>(Shape.Create(AttentionDim), 0UL, 0UL) - 0.5.AsScalar<T>()));

            Softmax = Variable<T>();
            AttentionState = Variable<T>(PartialShape.Create(Batch, EncoderHiddenSize));
        }

        public override void Forward(Executor executor)
        {
            var wh = executor.GetTensor(Wh);
            var wd = executor.GetTensor(Wd);
            var v = executor.GetTensor(V);
            var h = executor.GetTensor(EncoderHiddenStates).Reshape(SeqLength*Batch, -1);
            var d = executor.GetTensor(DecoderHiddenState);

            var whh = Dot(h, wh);       // [n*b, EncoderHiddenSize] * [EncoderHiddenSize, AttentionDim] = [n*b, AttentionDim]
            var wdd = Dot(d, wd);       // [b, DecoderHiddenSize] * [DecoderHiddenSize, AttentionDim] = [b, AttentionDim]
            var whd = Tanh(whh + wdd);  // broadcasting to [n*b, AttentionDim]

            var u = Dot(whd, v);        // [n*b, AttentionDim] * [AttentionDim] = [n*b]

            var expu = Exp(u.Reshape(SeqLength, Batch));
            var softmax = expu/ReduceSum(expu, true, 0);  // [n, b]
            executor.AssignTensor(Softmax, softmax);

            var ctx = executor.Context;
            if (ctx.Type == ContextType.Gpu && typeof(T) == typeof(float))
            {
                var stream = ctx.ToGpuContext().Stream;
                var hPtr = h.Buffer.Ptr.Reinterpret<float>();
                var softmaxPtr = executor.GetTensor(Softmax).Buffer.Ptr.Reinterpret<float>();
                var attentionState = executor.GetTensor(AttentionState).Buffer.Ptr.Reinterpret<float>();

                var batchSize = Batch;
                var seqLength = SeqLength;
                var encoderHiddenSize = EncoderHiddenSize;

                // strides for hPtr: [n*b, b, 1]
                // TODO proper size
                var lp = new LaunchParam(new dim3(batchSize/32, encoderHiddenSize/32, 1), new dim3(32, 32));
                stream.Launch(() =>
                {
                    var batch = blockIdx.x*blockDim.x + threadIdx.x;
                    var hidden = blockIdx.y*blockDim.y + threadIdx.y;
                    if (batch < batchSize && hidden < EncoderHiddenSize)
                    {
                        var sum = 0.0f;
                        for (var i = 0; i < seqLength; ++i)
                        {
                            var alpha = softmaxPtr[i * batchSize + batch];
                            sum += alpha * hPtr[i * seqLength * batchSize + batch * batchSize + hidden];
                        }
                        attentionState[batch * encoderHiddenSize + hidden] = sum;
                    }
                }, lp);
            }
            else
            {
                throw new NotImplementedException();
            }
        }

        public override void Backward(Executor executor)
        {
            throw new NotImplementedException();
        }
    }

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
