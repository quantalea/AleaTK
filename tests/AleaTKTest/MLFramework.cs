using System;
using System.Linq;
using AleaTK;
using AleaTK.ML;
using AleaTK.ML.Operator;
using NUnit.Framework;
using static AleaTK.Library;
using static AleaTK.ML.Library;
using static AleaTKUtil.Common;
using static AleaTKTest.Common;
using Executor = AleaTK.ML.Executor;

namespace AleaTKTest
{
    public static class MLFramework
    {
        public class ImperativeLoopOp : Differentiable
        {
            public int SeqLength { get; }
            public long Size { get; }

            public Variable<float> Input { get; }
            public Variable<float> Weight { get; }
            public Variable<float> Output { get; }
            public Variable<float> Temp { get; }

            public Variable<float> SubInput { get; }
            public Variable<float> SubWeight { get; }
            public Variable<float> SubOutput { get; }

            public Symbol SubExecutor { get; } = new Symbol();

            public ImperativeLoopOp(Variable<float> input, int seqLength)
            {
                SeqLength = seqLength;

                Input = input;
                var size = input.Shape[1];
                Size = size;

                Weight = Parameter<float>(Fill(Shape.Create(seqLength, size, size), 1.0f));

                Output = Variable<float>(PartialShape.Create(size, size));

                Temp = AuxVariable<float>();

                AddInput(Input);
                AddInput(Weight);
                AddOutput(Output);
                AddAuxVar(Temp);

                SubInput = Variable<float>(input.Shape);
                SubWeight = Variable<float>(input.Shape);
                SubOutput = Dot(SubInput, SubWeight);
            }

            public override void Initialize(Executor executor)
            {
                var subExecutor = new Executor(executor.Context, SubOutput);
                executor.Objects[SubExecutor] = subExecutor;
                subExecutor.Initalize();
                base.Initialize(executor);
            }

            public override void Forward(Executor executor)
            {
                var subExecutor = (Executor)executor.Objects[SubExecutor];
                var input = executor.GetTensor(Input);
                var output = executor.GetTensor(Output, Shape.Create(Size, Size));
                var temp = executor.GetTensor(Temp, Shape.Create(SeqLength - 1, Size, Size));

                for (var t = 0; t < SeqLength; ++t)
                {
                    
                }


                throw new NotImplementedException();
            }

            public override void Backward(Executor executor)
            {
                throw new NotImplementedException();
            }
        }

        [Test]
        public static void TestImperativeLoopOp()
        {
            var ctx = Context.GpuContext(0);

            var size = 100;
            var input = Variable<float>(PartialShape.Create(size, size));
            var loop = new ImperativeLoopOp(input, size);

            var exe = new Executor(ctx, loop.Output);
            exe.Initalize();

            var inputHost = new float[size, size];
            var rng = new Random();
            for (var i = 0; i < size; ++i)
            {
                for (var j = 0; j < size; ++j)
                {
                    inputHost[i, j] = (float)rng.NextDouble();
                }
            }

            exe.AssignTensor(loop.Input, inputHost.AsTensor());
            exe.Forward();


        }

        public class Attention<T>
        {
            public Variable<T> EncoderHiddenStates { get; }
            public Variable<T> DecoderHiddenStates { get; }

            public Variable<T> Wh { get; }
            public Variable<T> Wd { get; }
            public Variable<T> V { get; }

            public long AttentionDim { get; }
            public long EncoderHiddenSize { get; }
            public long DecoderHiddenSize { get; }
            public long Batch { get; }

            public Variable<T> Output { get; }

            public Attention(Variable<T> encoderHiddenStates, Variable<T> decoderHiddenStates, long attentionDim)
            {
                AttentionDim = attentionDim;
                EncoderHiddenStates = encoderHiddenStates;
                DecoderHiddenStates = decoderHiddenStates;

                // one goal is, try to make batchSize and encoderSeqLength unknown at symbol layer
                // so, in LSTM outer op, we can create one graph and one sub-executor, and applied for
                // different encoderSeqLength and batchSize.
                Util.EnsureEqual(3, EncoderHiddenStates.Shape.Rank, "states layout: (encoderSeqLength, batch, encoderHiddenSize)");
                Util.EnsureTrue(EncoderHiddenStates.Shape[2] > 0, "states should be determined.");
                EncoderHiddenSize = EncoderHiddenStates.Shape[2];

                Util.EnsureEqual(2, DecoderHiddenStates.Shape.Rank, "DecoderHiddenStates layout: (batch, decoderHiddenSize)");
                Util.EnsureTrue(DecoderHiddenStates.Shape[1] > 0, "DecoderHiddenStates should be determined.");
                DecoderHiddenSize = DecoderHiddenStates.Shape[1];

                var scaleWh = Sqrt(12.0.AsScalar<T>() / ((double)(AttentionDim + EncoderHiddenSize)).AsScalar<T>());
                Wh = Parameter(scaleWh * (RandomUniform<T>(Shape.Create(EncoderHiddenSize, AttentionDim), 0UL, 0UL) - 0.5.AsScalar<T>()));

                var scaleWd = Sqrt(12.0.AsScalar<T>() / ((double)(AttentionDim + DecoderHiddenSize)).AsScalar<T>());
                Wd = Parameter(scaleWd * (RandomUniform<T>(Shape.Create(DecoderHiddenSize, AttentionDim), 0UL, 0UL) - 0.5.AsScalar<T>()));

                var scaleV = Sqrt(12.0.AsScalar<T>() / ((double)(AttentionDim)).AsScalar<T>());
                V = Parameter(scaleV * (RandomUniform<T>(Shape.Create(AttentionDim, 1), 0UL, 0UL) - 0.5.AsScalar<T>()));

                // build the graph
                var h = EncoderHiddenStates.Reshape(-1, EncoderHiddenSize); // (n*b,He) He denotes hiddenSize of encoder
                var d = DecoderHiddenStates; // (b,Hd) Hd denotes hiddenSize of decoder
                var whh = Dot(h, Wh); // shape (n*b,K) K denotes attentionDim
                var wdd = Dot(d, Wd); // shape (b,K)

                // to add whh and wdd, we need broadcast, for this, we need to know at least n or b.
                // The decision here is to make b known at symbolic layer, because then you can have
                // flexibility on n (EncoderSeqLength), easier for making bucket.
                // another issue is, our backward of add has some issue dealing with 3d array which has broadcast
                // so, we can reshape them into 2d tensor here:
                // initial shape: (n*b,K) + (b,K)
                // reshape for the boadcast: (n,b*K) + (b*K) and for broadcasting (b*K) will broadcast to (1,b*K)
                // then: (n,b*K) + (b*K) = (n,b*K)
                // reshape result to (n*b,K)
                Batch = EncoderHiddenStates.Shape[1];
                Util.EnsureTrue(Batch > 0, "Batch need to be determined.");
                Util.EnsureTrue(Batch == DecoderHiddenStates.Shape[0]);
                var sum = (whh.Reshape(-1, Batch * AttentionDim) + wdd.Reshape(-1)).Reshape(-1, AttentionDim);

                // tanh, shape no change (n*b,K)
                var whd = new ActivationTanh<T>(sum);

                // (n*b,K) dot (K,1) = (n*b,1) => reshape to (n,b)
                var u = Dot(whd.Output, V).Reshape(-1, Batch);

                // same shape (n,b)
                var softmax = new Softmax<T>(u);

                Output = softmax.Output;
            }
        }

        [Test]
        public static void TestAttention()
        {
            var batch = 4;
            var encoderHiddenSize = 5;
            var decoderHiddenSize = 4;
            var attentionDim = 3;

            // (encoderSeqLength, batch, encoderHiddenSize)
            var encoderHiddenStates = Variable<float>(PartialShape.Create(-1, batch, encoderHiddenSize));
            var decoderHiddenStates = Variable<float>(PartialShape.Create(batch, decoderHiddenSize));
            var attention = new Attention<float>(encoderHiddenStates, decoderHiddenStates, attentionDim);

            var ctx = Context.GpuContext(0);
            var exe = new Executor(ctx, attention.Output) { AssignAllGradient = true };
            exe.Initalize();

            // encoderSeqLength is flexibly at runtime
            var encoderSeqLength = 3;
            var dataEncoderHiddenStates = new float[encoderSeqLength, batch, encoderHiddenSize];
            AleaTKUtil.Common.UniformRandomArray(dataEncoderHiddenStates);

            var dataDecoderHiddenStates = new float[batch, decoderHiddenSize];
            UniformRandomArray(dataDecoderHiddenStates);

            exe.AssignTensor(encoderHiddenStates, dataEncoderHiddenStates.AsTensor());
            exe.AssignTensor(decoderHiddenStates, dataDecoderHiddenStates.AsTensor());
            exe.Forward();

            var tensorOutput = exe.GetTensor(attention.Output);
            Console.WriteLine(tensorOutput.Shape);
            tensorOutput.Reshape(encoderSeqLength * batch, -1).Print();

            var dataDOutput = new float[encoderSeqLength, batch];
            UniformRandomArray(dataDOutput);
            exe.AssignGradientDirectly(attention.Output, dataDOutput.AsTensor());
            exe.Backward();

            var tensorDWh = exe.GetGradient(attention.Wh);
            tensorDWh.Print();

            var tensorDWd = exe.GetGradient(attention.Wd);
            tensorDWd.Print();

            var tensorDH = exe.GetGradient(attention.EncoderHiddenStates);
            Console.WriteLine(tensorDH.Shape);
            tensorDH.Reshape(-1, encoderHiddenSize).Print();

            var tensorDD = exe.GetGradient(attention.DecoderHiddenStates);
            Console.WriteLine(tensorDD.Shape);
            tensorDD.Print();
        }

        /// <summary>
        /// Calculates the weighted sum
        /// 
        ///     c_{b, k} = \sum_{i = 0}^n a_i h_{i, b, k} 
        ///  
        /// with weight vector a \in \mathbb{R}^n.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        public class WeightedSumReduce<T> : Differentiable
        {
            public WeightedSumReduce(Variable<T> weights, Variable<T> vectors)
            {
                Weights = weights;
                Vectors = vectors;

                // output shape is the broadcast of w*v, skip the first dimension
                // currently the reduce is fixed on first dimension
                if (weights.HasShape && vectors.HasShape)
                {
                    var shape = PartialShape.Broadcast(weights.Shape, vectors.Shape);
                    Output = Variable<T>(PartialShape.Create(shape.Skip(1).ToArray()));
                }
                else
                {
                    Output = Variable<T>();
                }
                //Output = Variable<T>();

                AddInput(Weights);
                AddInput(Vectors);
                AddOutput(Output);
            }

            public Variable<T> Weights { get; }

            public Variable<T> Vectors { get; }

            public Variable<T> Output { get; }

            public override void Forward(Executor executor)
            {
                var vectors = executor.GetTensor(Vectors);
                var weights = executor.GetTensor(Weights);
                
                // the result of prod need to be reshape to matrix for later to do dot
                var l0 = Math.Max(weights.Shape[0], vectors.Shape[0]);
                var prod = (weights*vectors).Reshape(l0, -1);

                // after dot, it will reshape back to what they should be
                var shape = Shape.Broadcast(weights.Shape, vectors.Shape).Skip(1).ToArray();
                var reduce = ReduceSum(prod, 0).Reshape(shape);

                executor.AssignTensor(Output, reduce);
            }

            public override void Backward(Executor executor)
            {
                var vectors = executor.GetTensor(Vectors);
                var weights = executor.GetTensor(Weights);
                var dOutput = executor.GetGradient(Output);
                
                //var n = states.Shape[0];
                //var b = Batch;
                //var d = VectorSize;

                var l0 = Math.Max(weights.Shape[0], vectors.Shape[0]);

                // issue:
                // v.shape -> (n, b, d)
                // w.shape -> (n, b, 1)
                // NOTE, the weight here has no meaning, it might be NOT learnable!
                // o.shape -> (b, d)
                // the dv, dw, and do shape are same as there tensor shape
                // so v dot dO => (n, b*d) dot (b*d, 1) => (n,1)
                // w dot dO    => (n, b) dot (b*d, 1) => ???


                //var dWeights = Dot(states.Reshape(-1, b*d), dOutput.Reshape(b*d, -1)).Reshape(n);
                //var dVectors = Dot(softmax.Reshape(n, 1), dOutput.Reshape(1, b*d)).Reshape(n, b, d);
                //executor.AssignGradient(softmax, dWeights);
                //executor.AssignGradient(states, dVectors);

            }
        }

        public class AttentionReduce<T> : Differentiable
        {
            public AttentionReduce(Variable<T> softmax, Variable<T> states)
            {
                Softmax = softmax;
                States = states;

                Util.EnsureTrue(softmax.Shape.Rank == 2, "Softmax: (n,b)");
                Util.EnsureTrue(states.Shape.Rank == 3, "States: (n,b,d)");
                Util.EnsureTrue(softmax.Shape[1] > 0, "Softmax: b needed.");
                Util.EnsureTrue(states.Shape[1] > 0, "States: b needed.");
                Util.EnsureTrue(states.Shape[2] > 0, "States: d needed.");
                Util.EnsureTrue(softmax.Shape[1] == states.Shape[1], "b should match.");

                BatchSize = softmax.Shape[1];
                StatesSize = states.Shape[2];

                Output = Variable<T>(PartialShape.Create(BatchSize, StatesSize));

                AddInput(Softmax);
                AddInput(States);
                AddOutput(Output);
            }

            public Variable<T> Softmax { get; }

            public Variable<T> States { get; }

            public Variable<T> Output { get; }

            public long BatchSize { get; }

            public long StatesSize { get; }

            public override void Forward(Executor executor)
            {
                var states = executor.GetTensor(States);
                var softmax = executor.GetTensor(Softmax);
                var n = states.Shape[0];
                var b = states.Shape[1];
                var d = states.Shape[2];

                var prod = softmax.Reshape(n, b, 1)*states;

                // currently reduce sum only works up to 2d tensor
                // then we do a reduce to make it an 2d tensor
                // after reduce, we reshape it back.
                var reduce = ReduceSum(prod.Reshape(n, b*d), 0).Reshape(b, d);

                executor.AssignTensor(Output, reduce);
            }

            public override void Backward(Executor executor)
            {
                var states = executor.GetTensor(States);
                var softmax = executor.GetTensor(Softmax);
                var dOutput = executor.GetGradient(Output);
                var n = states.Shape[0];
                var b = states.Shape[1];
                var d = states.Shape[2];

                executor.AssignGradient(States, softmax.Reshape(n,b,1)*dOutput);

                // states (n,b,d) * dOutput (b,d) => (n,b,d)
                // softmax (n,b)
                executor.AssignGradient(Softmax, ReduceSum((states*dOutput).Reshape(n*b, d), 1).Reshape(n, b));
            }
        }

        [Test]
        public static void TestWeightedReductionEvaluator()
        {
            var n = 3;
            var b = 4;
            var d = 5;

            var statesData = new double[n, b, d];
            UniformRandomArray(statesData);
            var softmaxData = new double[n, b];
            UniformRandomArray(softmaxData);

            var softmax = Variable<double>(PartialShape.Create(-1, b));
            var states = Variable<double>(PartialShape.Create(-1, b, d));
            var reduce = new AttentionReduce<double>(softmax, states);

            var ctx = Context.GpuContext(0);
            var exe = new Executor(ctx, reduce.Output) { AssignAllGradient = true };
            exe.Initalize();

            var dOutputData = new double[b, d];
            UniformRandomArray(dOutputData);

            exe.AssignTensor(softmax, softmaxData.AsTensor());
            exe.AssignTensor(states, statesData.AsTensor());
            exe.Forward();
            exe.AssignGradientDirectly(reduce.Output, dOutputData.AsTensor());
            exe.Backward();

            var dSoftmax = exe.GetGradient(reduce.Softmax);
            var dStates = exe.GetGradient(reduce.States);

            var bump = 1e-6;

            var dSoftmaxFd = GradientChecker.FiniteDifferenceGradient(exe, softmax, bump: bump);
            AreClose(dSoftmaxFd.ToArray2D(), dSoftmax.ToArray2D(), 1e-7);

            var dStatesFd = GradientChecker.FiniteDifferenceGradient(exe, states, bump: bump);
            AreClose(dStatesFd.ToArray3D(), dStates.ToArray3D(), 1e-7);

            //var dVectorsFdArray = dVectorsFd.Reshape(-1).ToArray();
            //var dVectorsBackpropArray = dStates.Reshape(-1).ToArray();
            //var err = MaxAbsDiff(dVectorsFdArray, dVectorsBackpropArray);
        }

        public class Attention2<T>
        {
            public Variable<T> EncoderHiddenStates { get; }
            public Variable<T> DecoderHiddenStates { get; }

            public Variable<T> Wh { get; }
            public Variable<T> Wd { get; }
            public Variable<T> V { get; }

            public long AttentionDim { get; }
            public long EncoderHiddenSize { get; }
            public long DecoderHiddenSize { get; }
            public long Batch { get; }

            public Variable<T> Output { get; }

            public Attention2(Variable<T> encoderHiddenStates, Variable<T> decoderHiddenStates, long attentionDim)
            {
                AttentionDim = attentionDim;
                EncoderHiddenStates = encoderHiddenStates;
                DecoderHiddenStates = decoderHiddenStates;

                // one goal is, try to make batchSize and encoderSeqLength unknown at symbol layer
                // so, in LSTM outer op, we can create one graph and one sub-executor, and applied for
                // different encoderSeqLength and batchSize.
                Util.EnsureEqual(3, EncoderHiddenStates.Shape.Rank, "states layout: (encoderSeqLength, batch, encoderHiddenSize)");
                Util.EnsureTrue(EncoderHiddenStates.Shape[2] > 0, "states should be determined.");
                EncoderHiddenSize = EncoderHiddenStates.Shape[2];

                Util.EnsureEqual(2, DecoderHiddenStates.Shape.Rank, "DecoderHiddenStates layout: (batch, decoderHiddenSize)");
                Util.EnsureTrue(DecoderHiddenStates.Shape[1] > 0, "DecoderHiddenStates should be determined.");
                DecoderHiddenSize = DecoderHiddenStates.Shape[1];

                var scaleWh = Sqrt(12.0.AsScalar<T>() / ((double)(AttentionDim + EncoderHiddenSize)).AsScalar<T>());
                Wh = Parameter(scaleWh * (RandomUniform<T>(Shape.Create(EncoderHiddenSize, AttentionDim), 0UL, 0UL) - 0.5.AsScalar<T>()));

                var scaleWd = Sqrt(12.0.AsScalar<T>() / ((double)(AttentionDim + DecoderHiddenSize)).AsScalar<T>());
                Wd = Parameter(scaleWd * (RandomUniform<T>(Shape.Create(DecoderHiddenSize, AttentionDim), 0UL, 0UL) - 0.5.AsScalar<T>()));

                var scaleV = Sqrt(12.0.AsScalar<T>() / ((double)(AttentionDim)).AsScalar<T>());
                V = Parameter(scaleV * (RandomUniform<T>(Shape.Create(AttentionDim, 1), 0UL, 0UL) - 0.5.AsScalar<T>()));

                // build the graph
                var h = EncoderHiddenStates.Reshape(-1, EncoderHiddenSize); // (n*b,He) // He denotes hiddenSize of encoder
                var d = DecoderHiddenStates; // (b,Hd) // Hd denotes hiddenSize of decoder
                var whh = Dot(h, Wh); // shape (n*b,K) K denotes attentionDim
                var wdd = Dot(d, Wd); // shape (b,K)

                // to add whh and wdd, we need broadcast, for this, we need to know at least n or b.
                // The decision here is to make b known at symbolic layer, because then you can have
                // flexibility on n (EncoderSeqLength), easier for making bucket.
                // another issue is, our backward of add has some issue dealing with 3d array which has broadcast
                // so, we can reshape them into 2d tensor here:
                // initial shape: (n*b,K) + (b,K)
                // reshape for the boadcast: (n,b*K) + (b*K) (for broadcasting, (b*K) will broadcast to (1,b*K)
                // then: (n,b*K) + (b*K) = (n,b*K)
                // reshape result to (n*b,K)
                Batch = EncoderHiddenStates.Shape[1];
                Util.EnsureTrue(Batch > 0, "Batch need to be determined.");
                Util.EnsureTrue(Batch == DecoderHiddenStates.Shape[0]);
                var add = (whh.Reshape(-1, Batch * AttentionDim) + wdd.Reshape(-1)).Reshape(-1, AttentionDim);

                // tanh, shape no change (n*b,K)
                var whd = new ActivationTanh<T>(add);

                // (n*b,K) dot (K,1) = (n*b,1) => reshape to (n,b)
                var u = Dot(whd.Output, V).Reshape(-1, Batch);

                // same shape (n,b)
                var softmax = new Softmax<T>(u);

                // sum (n,b) * (n,b,d)
                var reduce = new AttentionReduce<T>(softmax.Output.Reshape(-1, Batch), EncoderHiddenStates);

                Output = reduce.Output;
            }
        }

        [Test]
        public static void TestAttention2()
        {
            var batch = 4;
            var encoderHiddenSize = 5;
            var decoderHiddenSize = 4;
            var attentionDim = 3;

            // (encoderSeqLength, batch, encoderHiddenSize)
            var encoderHiddenStates = Variable<double>(PartialShape.Create(-1, batch, encoderHiddenSize));
            var decoderHiddenStates = Variable<double>(PartialShape.Create(batch, decoderHiddenSize));
            var attention = new Attention2<double>(encoderHiddenStates, decoderHiddenStates, attentionDim);

            var ctx = Context.GpuContext(0);
            var exe = new Executor(ctx, attention.Output) { AssignAllGradient = true };
            exe.Initalize();

            // encoderSeqLength is flexibly at runtime
            var encoderSeqLength = 3;
            var dataEncoderHiddenStates = new double[encoderSeqLength, batch, encoderHiddenSize];
            UniformRandomArray(dataEncoderHiddenStates);

            var dataDecoderHiddenStates = new double[batch, decoderHiddenSize];
            UniformRandomArray(dataDecoderHiddenStates);

            exe.AssignTensor(encoderHiddenStates, dataEncoderHiddenStates.AsTensor());
            exe.AssignTensor(decoderHiddenStates, dataDecoderHiddenStates.AsTensor());
            exe.Forward();

            var tensorOutput = exe.GetTensor(attention.Output);
            //Console.WriteLine(tensorOutput.Shape);
            //tensorOutput.Print();

            var dataDOutput = new double[batch, encoderHiddenSize];
            UniformRandomArray(dataDOutput);
            exe.AssignGradientDirectly(attention.Output, dataDOutput.AsTensor());
            exe.Backward();

            var tensorDWh = exe.GetGradient(attention.Wh);
            //tensorDWh.Print();

            var tensorDWd = exe.GetGradient(attention.Wd);
            //tensorDWd.Print();

            var tensorDH = exe.GetGradient(attention.EncoderHiddenStates);
            //Console.WriteLine(tensorDH.Shape);
            //tensorDH.Reshape(-1, encoderHiddenSize).Print();

            var tensorDD = exe.GetGradient(attention.DecoderHiddenStates);
            //Console.WriteLine(tensorDD.Shape);
            //tensorDD.Print();

            var bump = 1e-7;

            var tensorDWh_fd = GradientChecker.FiniteDifferenceGradient(exe, attention.Wh, bump: bump);
            //tensorDWh.Print();
            //tensorDWh_fd.Print();
            AreClose(tensorDWh.ToArray2D(), tensorDWh_fd.ToArray2D(), 1e-7);

            var tensorDWd_fd = GradientChecker.FiniteDifferenceGradient(exe, attention.Wd, bump: bump);
            //tensorDWd.Print();
            //tensorDWd_fd.Print();
            AreClose(tensorDWd.ToArray2D(), tensorDWd_fd.ToArray2D(), 1e-7);

            var tensorDH_fd = GradientChecker.FiniteDifferenceGradient(exe, attention.EncoderHiddenStates, bump: bump);
            //tensorDH.Reshape(-1, encoderHiddenSize).Print();
            //tensorDH_fd.Reshape(-1, encoderHiddenSize).Print();
            // TODO: bug here, not match

            var tensorDD_fd = GradientChecker.FiniteDifferenceGradient(exe, attention.DecoderHiddenStates, bump: bump);
            //tensorDD.Print();
            //tensorDD_fd.Print();
            AreClose(tensorDD.ToArray2D(), tensorDD_fd.ToArray2D(), 1e-7);
        }
    }
}
