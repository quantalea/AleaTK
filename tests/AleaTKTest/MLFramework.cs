using System;
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
                Util.EnsureEqual(3, EncoderHiddenStates.Shape.Rank, "Vectors layout: (encoderSeqLength, batch, encoderHiddenSize)");
                Util.EnsureTrue(EncoderHiddenStates.Shape[2] > 0, "Vectors should be determined.");
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
                this.Vectors = vectors;
                Batch = this.Vectors.Shape[1];
                VectorSize = this.Vectors.Shape[2];
                Output = Variable<T>(PartialShape.Create(Batch, VectorSize));

                AddInput(Weights);
                AddInput(this.Vectors);
                AddOutput(Output);
            }

            public Variable<T> Weights { get; }

            public Variable<T> Vectors { get; }

            public Variable<T> Output { get; }

            public long Batch { get; }

            public long VectorSize { get; }

            public override void Forward(Executor executor)
            {
                var vectors = executor.GetTensor(Vectors);
                var weights = executor.GetTensor(Weights);
                var n = vectors.Shape[0];
                var b = Batch;
                var d = VectorSize;

                var prod = (weights.Reshape(n, 1)*vectors.Reshape(n, b*d)).Reshape(n, b*d);
                var reduce = ReduceSum(prod, 0).Reshape(b, d);
                executor.AssignTensor(Output, reduce);
            }

            public override void Backward(Executor executor)
            {
                var vectors = executor.GetTensor(Vectors);
                var weights = executor.GetTensor(Weights);
                var n = vectors.Shape[0];
                var b = Batch;
                var d = VectorSize;

                var dOutput = executor.GetGradient(Output);
                var dWeights = Dot(vectors.Reshape(-1, b*d), dOutput.Reshape(b*d, -1)).Reshape(n);
                var dVectors = Dot(weights.Reshape(n, 1), dOutput.Reshape(1, b*d)).Reshape(n, b, d);
                executor.AssignGradient(Weights, dWeights);
                executor.AssignGradient(Vectors, dVectors);
            }
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
                Util.EnsureEqual(3, EncoderHiddenStates.Shape.Rank, "Vectors layout: (encoderSeqLength, batch, encoderHiddenSize)");
                Util.EnsureTrue(EncoderHiddenStates.Shape[2] > 0, "Vectors should be determined.");
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

                var reduce = new WeightedSumReduce<T>(softmax.Output, EncoderHiddenStates);

                Output = reduce.Output;
            }
        }

        [Test]
        public static void TestWeightedReductionEvaluator()
        {
            var seqLength = 3;
            var batch = 4;
            var vectorSize = 5;

            var vectorsData = new float[seqLength, batch, vectorSize];
            UniformRandomArray(vectorsData);
            var weightsData = new float[seqLength];
            UniformRandomArray(weightsData);

            var weights = Variable<float>(PartialShape.Create(seqLength));
            var vectors = Variable<float>(PartialShape.Create(-1, batch, vectorSize));
            var weightedReduce = new WeightedSumReduce<float>(weights, vectors);

            var ctx = Context.GpuContext(0);
            var exe = new Executor(ctx, weightedReduce.Output) { AssignAllGradient = true };
            exe.Initalize();

            var dOutputData = new float[batch, vectorSize];
            UniformRandomArray(dOutputData);

            exe.AssignTensor(weights, weightsData.AsTensor());
            exe.AssignTensor(vectors, vectorsData.AsTensor());
            exe.Forward();
            exe.AssignGradientDirectly(weightedReduce.Output, dOutputData.AsTensor());
            exe.Backward();

            var dWeights = exe.GetGradient(weightedReduce.Weights);
            var dVectors = exe.GetGradient(weightedReduce.Vectors);

            var dWeightsFd = GradientChecker.FiniteDifferenceGradient(exe, weights, weightedReduce.Output);
            AreClose(dWeightsFd, dWeights, 1e-2);

            var dVectorsFd = GradientChecker.FiniteDifferenceGradient(exe, vectors, weightedReduce.Output);
            AreClose(dVectorsFd, dVectors, 0.005);

            var dVectorsFdArray = dVectorsFd.Reshape(-1).ToArray();
            var dVectorsBackpropArray = dVectors.Reshape(-1).ToArray();
            var err = MaxAbsDiff(dVectorsFdArray, dVectorsBackpropArray);
        }

        [Test]
        public static void TestAttention2()
        {
            var batch = 4;
            var encoderHiddenSize = 5;
            var decoderHiddenSize = 4;
            var attentionDim = 3;

            // (encoderSeqLength, batch, encoderHiddenSize)
            var encoderHiddenStates = Variable<float>(PartialShape.Create(-1, batch, encoderHiddenSize));
            var decoderHiddenStates = Variable<float>(PartialShape.Create(batch, decoderHiddenSize));
            var attention = new Attention2<float>(encoderHiddenStates, decoderHiddenStates, attentionDim);

            var ctx = Context.GpuContext(0);
            var exe = new Executor(ctx, attention.Output) { AssignAllGradient = true };
            exe.Initalize();

            // encoderSeqLength is flexibly at runtime
            var encoderSeqLength = 3;
            var dataEncoderHiddenStates = new float[encoderSeqLength, batch, encoderHiddenSize];
            UniformRandomArray(dataEncoderHiddenStates);

            var dataDecoderHiddenStates = new float[batch, decoderHiddenSize];
            UniformRandomArray(dataDecoderHiddenStates);

            exe.AssignTensor(encoderHiddenStates, dataEncoderHiddenStates.AsTensor());
            exe.AssignTensor(decoderHiddenStates, dataDecoderHiddenStates.AsTensor());
            exe.Forward();

            var tensorOutput = exe.GetTensor(attention.Output);
            Console.WriteLine(tensorOutput.Shape);
            tensorOutput.Print();

            //var dataDOutput = new float[encoderSeqLength, batch];
            //UniformRandomArray(dataDOutput);
            //exe.AssignGradientDirectly(attention.Output, dataDOutput.AsTensor());
            //exe.Backward();

            //var tensorDWh = exe.GetGradient(attention.Wh);
            //tensorDWh.Print();

            //var tensorDWd = exe.GetGradient(attention.Wd);
            //tensorDWd.Print();

            //var tensorDH = exe.GetGradient(attention.Vectors);
            //Console.WriteLine(tensorDH.Shape);
            //tensorDH.Reshape(-1, encoderHiddenSize).Print();

            //var tensorDD = exe.GetGradient(attention.DecoderHiddenStates);
            //Console.WriteLine(tensorDD.Shape);
            //tensorDD.Print();
        }
    }
}
