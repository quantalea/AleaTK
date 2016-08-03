using System;
using System.Linq;
using System.Runtime.ConstrainedExecution;
using Alea;
using Alea.cuDNN;
using Alea.Parallel;
using static AleaTK.Library;
using static AleaTK.ML.Library;

namespace AleaTK.ML.Operator
{
    public class RnnDynamicDescr<T>
    {
        public RnnDynamicDescr(Executor executor, RnnDynamic<T> rnn)
        {
            var context = executor.Context.ToGpuContext();
            var dnn = context.Dnn;

            Rnn = rnn;
            var x = executor.GetTensor(Rnn.X);

            SeqLength = (int)x.Shape[0];
            MiniBatch = (int)x.Shape[1];

            var shape = Shape.Create(SeqLength, MiniBatch, Rnn.HiddenSize);
            executor.GetTensor(Rnn.Y, shape);

            // state variables
            shape = Shape.Create(Rnn.NumLayers, MiniBatch, Rnn.HiddenSize);
            var strides = Strides.Create(shape[1] * shape[2], shape[2], 1); // inner change most
            executor.GetTensor(Rnn.HX, shape);
            executor.GetTensor(Rnn.CX, shape);
            executor.GetTensor(Rnn.HY, shape);
            executor.GetTensor(Rnn.CY, shape);
            StateDesc = new TensorDescriptor();
            StateDesc.SetND(Dnn.DataTypeOf<T>(), shape.AsInt32Array, strides.AsInt32Array);

            // xDesc is an array, for each step
            shape = Shape.Create(MiniBatch, rnn.InputSize, 1);
            strides = Strides.Create(shape[1] * shape[2], shape[2], 1);
            var xDesc = new TensorDescriptor();
            xDesc.SetND(Dnn.DataTypeOf<T>(), shape.AsInt32Array, strides.AsInt32Array);
            XDesc = Enumerable.Repeat(xDesc, SeqLength).ToArray();

            // yDesc is an array, for each step
            shape = Shape.Create(MiniBatch, rnn.HiddenSize, 1);
            strides = Strides.Create(shape[1] * shape[2], shape[2], 1);
            var yDesc = new TensorDescriptor();
            yDesc.SetND(Dnn.DataTypeOf<T>(), shape.AsInt32Array, strides.AsInt32Array);
            YDesc = Enumerable.Repeat(yDesc, SeqLength).ToArray();

            // workspace and reserved space
            var rnnDesc = executor.RnnDescDict[rnn.RnnDesc];
            IntPtr workSize;
            dnn.GetRNNWorkspaceSize(rnnDesc, SeqLength, XDesc, out workSize);
            executor.GetTensor(Rnn.Workspace, Shape.Create(workSize.ToInt64()));

            if (Rnn.IsTraining)
            {
                IntPtr reserveSize;
                dnn.GetRNNTrainingReserveSize(rnnDesc, SeqLength, XDesc, out reserveSize);
                executor.GetTensor(Rnn.ReserveSpace, Shape.Create(reserveSize.ToInt64()));

                executor.GetGradient(Rnn.X, Shape.Create(Rnn.X.Shape.AsArray));
                executor.GetGradient(Rnn.Y, Shape.Create(Rnn.Y.Shape.AsArray));
                executor.GetGradient(Rnn.HX, Shape.Create(Rnn.HX.Shape.AsArray));
                executor.GetGradient(Rnn.CX, Shape.Create(Rnn.CX.Shape.AsArray));
            }
        }

        public RnnDynamic<T> Rnn { get; }

        public int SeqLength { get; }

        public int MiniBatch { get; }

        public TensorDescriptor StateDesc { get; }

        public TensorDescriptor[] XDesc { get; }

        public TensorDescriptor[] YDesc { get; }

        public void AssignInitialStates(Executor executor, Tensor<T> hx, Tensor<T> cx)
        {
            executor.AssignTensor(Rnn.HX, hx);
            executor.AssignTensor(Rnn.CX, cx);
        }

        public void AssignInitialStates(Executor executor)
        {
            const double value = 0.0;
            executor.AssignTensor(Rnn.HX, Fill(Shape.Create(Rnn.HX.Shape.AsArray), ScalarOps.Conv<T>(value)));
            executor.AssignTensor(Rnn.CX, Fill(Shape.Create(Rnn.CX.Shape.AsArray), ScalarOps.Conv<T>(value)));
        }
    }

    /// <summary>
    /// Dynamically allocated recurrent neural network, possibly multiple layers stacked on each other, accelerated with cuDNN.
    /// This version can be with bucketed data of variable sequence and minibatch size.
    /// 
    /// Note that cuDNN adds dropout only between the layers, hence dropout for the input and output has to be added seperately.
    /// 
    /// Todo: implement other Rnn types such as GRU, RNN_RELU, RNN_TANH
    /// </summary>
    public class RnnDynamic<T> : Differentiable
    {
        public RnnDynamic(RnnType ty, Variable<T> x, int numLayers, int hiddenSize, bool isTraining = true, double dropout = 0.0, ulong dropoutSeed = 1337UL)
        {
            Type = ty;
            IsTraining = isTraining;
            NumLayers = numLayers;
            HiddenSize = hiddenSize;
            Dropout = isTraining ? dropout : 0.0;
            DropoutSeed = dropoutSeed;

            // X shape (seqLength, batch, inputSize)
            X = x;
            Util.EnsureEqual(3, X.Shape.Rank, "Input layout: (seqLength, batch, inputSize)");
            Util.EnsureTrue(X.Shape[2] >= 0, "Input layout: (seqLength, batch, inputSize)");
            InputSize = (int)X.Shape[2];

            // Y Shape (maxSeqLength, not yet known, hiddenSize)
            Y = Variable<T>(PartialShape.Create(-1, -1, HiddenSize));

            // W shape will be determined during initialization
            W = Parameter<T>();

            // state variables
            var shape = PartialShape.Create(NumLayers, -1, HiddenSize);
            HX = Variable<T>(shape);
            CX = Variable<T>(shape);
            HY = Variable<T>(shape);
            CY = Variable<T>(shape);

            // construct the graph
            AddInput(X);
            AddInput(W);
            AddOutput(Y);
            AddAuxVar(HX);
            AddAuxVar(CX);
            AddAuxVar(HY);
            AddAuxVar(CY);
            AddAuxVar(DropoutStates);
            AddAuxVar(Workspace);
            AddAuxVar(ReserveSpace);
        }

        public RnnType Type { get; }

        public bool IsTraining { get; }

        public double Dropout { get; }

        public ulong DropoutSeed { get; }

        public int NumLayers { get; }

        public int HiddenSize { get; }

        public int InputSize { get; }

        public Variable<T> X { get; }

        public Variable<T> HX { get; }

        public Variable<T> CX { get; }

        public Variable<T> Y { get; }

        public Variable<T> HY { get; }

        public Variable<T> CY { get; }

        public Variable<T> W { get; }

        public readonly Variable<byte> DropoutStates = Library.AuxVariable<byte>();

        public readonly Variable<byte> Workspace = Library.AuxVariable<byte>();

        public readonly Variable<byte> ReserveSpace = Library.AuxVariable<byte>();

        public readonly Symbol DropoutDesc = new Symbol();

        public readonly Symbol WDesc = new Symbol();

        public readonly Symbol RnnDesc = new Symbol();

        public readonly Symbol Descr = new Symbol();

        public override void Initialize(Executor executor)
        {
            var context = executor.Context.ToGpuContext();
            var dnn = context.Dnn;

            // dropout
            var dropoutDesc = executor.DropoutDescDict[DropoutDesc];
            IntPtr dropoutStatesSize;
            dnn.DropoutGetStatesSize(out dropoutStatesSize);
            var dropoutStates = executor.GetTensor(DropoutStates, Shape.Create(dropoutStatesSize.ToInt64()));
            dropoutDesc.Set(dnn, (float)Dropout, dropoutStates.Buffer.Ptr, dropoutStatesSize, DropoutSeed);

            // rnn descriptor
            var rnnDesc = executor.RnnDescDict[RnnDesc];
            var mode = Type.Mode;
            rnnDesc.Set(HiddenSize, NumLayers, dropoutDesc, RNNInputMode.LINEAR_INPUT, DirectionMode.UNIDIRECTIONAL, mode, Dnn.DataTypeOf<T>());

            // initialize weight, once only, using minibatch size 1
            var shape = PartialShape.Create(1, InputSize, 1); // first dimension does not affect the weight shape and size TODO test all, tested only for LSTM
            var strides = Strides.Create(shape[1] * shape[2], shape[2], 1);
            var xDesc = new TensorDescriptor();
            xDesc.SetND(Dnn.DataTypeOf<T>(), shape.AsInt32Array, strides.AsInt32Array);
            var wDesc = executor.FilterDescDict[WDesc];
            IntPtr weightsSize;
            dnn.GetRNNParamsSize(rnnDesc, xDesc, out weightsSize, Dnn.DataTypeOf<T>());
            Util.EnsureTrue(weightsSize.ToInt64() % Gpu.SizeOf<T>() == 0);
            var shapeW = Shape.Create(weightsSize.ToInt64() / Alea.Gpu.SizeOf<T>(), 1, 1);
            wDesc.SetND(Dnn.DataTypeOf<T>(), TensorFormat.CUDNN_TENSOR_NCHW, shapeW.AsInt32Array);

            // since we are using cuDNN, we'd better make sure these varaibles are allocated
            executor.GetTensor(W, shapeW);
            if (IsTraining) executor.GetGradient(W, shapeW);

            // init weights
            var numLinearLayers = Type.NumLinLayers;

            using (var filterDesc = new FilterDescriptor())
            {
                var w = executor.GetTensor(W);
                var filterDimA = new int[3];

                for (var layer = 0; layer < NumLayers; ++layer)
                {
                    for (var linLayerId = 0; linLayerId < numLinearLayers; ++linLayerId)
                    {
                        int nbDims;
                        DataType dataType;
                        TensorFormat format;

                        deviceptr<T> linLayerMat;
                        dnn.GetRNNLinLayerMatrixParams(rnnDesc, layer, xDesc, wDesc, w.Buffer.Ptr, linLayerId, filterDesc, out linLayerMat);

                        filterDesc.GetND(out dataType, out format, out nbDims, filterDimA);
                        var length = filterDimA.Aggregate(ScalarOps.Mul);

                        var linLayerMatBuffer = new Buffer<T>(context.Device, w.Memory, new Layout(Shape.Create(length)), linLayerMat);
                        var linLayerMatTensor = new Tensor<T>(linLayerMatBuffer);
                        context.Assign(linLayerMatTensor, RandomNormal<T>(Shape.Create(length)) / (Math.Sqrt(HiddenSize + InputSize).AsScalar<T>()));

                        deviceptr<T> linLayerBias;
                        dnn.GetRNNLinLayerBiasParams(rnnDesc, layer, xDesc, wDesc, w.Buffer.Ptr, linLayerId, filterDesc, out linLayerBias);

                        filterDesc.GetND(out dataType, out format, out nbDims, filterDimA);
                        length = filterDimA.Aggregate(ScalarOps.Mul);

                        var linLayerBiasBuffer = new Buffer<T>(context.Device, w.Memory, new Layout(Shape.Create(length)), linLayerBias);
                        var linLayerBiasTensor = new Tensor<T>(linLayerBiasBuffer);
                        Type.InitBias(context, layer, linLayerId, linLayerBiasTensor);
                    }
                }
            }

            base.Initialize(executor);
        }

        /// <summary>
        /// Call AssignInitialStates at least once before Forward or Backward. 
        /// </summary>
        /// <param name="executor"></param>
        /// <param name="hx"></param>
        /// <param name="cx"></param>
        public void AssignInitialStates(Executor executor, Tensor<T> hx, Tensor<T> cx)
        {
            var descr = (RnnDynamicDescr<T>) executor.Objects[Descr];
            descr.AssignInitialStates(executor, hx, cx);
        }

        /// <summary>
        /// Call AssignInitialStates at least once before Forward or Backward. 
        /// </summary>
        /// <param name="executor"></param>
        public void AssignInitialStates(Executor executor)
        {
            var descr = (RnnDynamicDescr<T>)executor.Objects[Descr];
            descr.AssignInitialStates(executor);
        }

        public override void Forward(Executor executor)
        {
            var descr = new RnnDynamicDescr<T>(executor, this);
            executor.Objects[Descr] = (object) descr;

            var context = executor.Context.ToGpuContext();
            var dnn = context.Dnn;
            var rnnDesc = executor.RnnDescDict[RnnDesc];
            var seqLength = descr.SeqLength;
            var xDesc = descr.XDesc;
            var x = executor.GetTensor(X);

            var hxDesc = descr.StateDesc;
            var hx = executor.GetTensor(HX);
            var cxDesc = descr.StateDesc;
            var cx = executor.GetTensor(CX);
            var wDesc = executor.FilterDescDict[WDesc];
            var w = executor.GetTensor(W);
            var yDesc = descr.YDesc;
            var y = executor.GetTensor(Y);
            var hyDesc = descr.StateDesc;
            var hy = executor.GetTensor(HY);
            var cyDesc = descr.StateDesc;
            var cy = executor.GetTensor(CY);
            var workspace = executor.GetTensor(Workspace);

            if (IsTraining)
            {
                var reserveSpace = executor.GetTensor(ReserveSpace);
                dnn.RNNForwardTraining(
                    rnnDesc, seqLength, xDesc, x.Buffer.Ptr, hxDesc, hx.Buffer.Ptr,
                    cxDesc, cx.Buffer.Ptr, wDesc, w.Buffer.Ptr, yDesc, y.Buffer.Ptr,
                    hyDesc, hy.Buffer.Ptr, cyDesc, cy.Buffer.Ptr,
                    workspace.Buffer.Ptr, (IntPtr)workspace.Shape.Length,
                    reserveSpace.Buffer.Ptr, (IntPtr)reserveSpace.Shape.Length);
            }
            else
            {
                dnn.RNNForwardInference(
                    rnnDesc, seqLength, xDesc, x.Buffer.Ptr, hxDesc, hx.Buffer.Ptr,
                    cxDesc, cx.Buffer.Ptr, wDesc, w.Buffer.Ptr, yDesc, y.Buffer.Ptr,
                    hyDesc, hy.Buffer.Ptr, cyDesc, cy.Buffer.Ptr,
                    workspace.Buffer.Ptr, (IntPtr)workspace.Shape.Length);
            }
        }

        public override void Backward(Executor executor)
        {
            var descr = (RnnDynamicDescr<T>)executor.Objects[Descr];

            Util.EnsureTrue(IsTraining);

            var context = executor.Context.ToGpuContext();
            var dnn = context.Dnn;

            if (executor.GetData(X).GradientAggregationCounter != 0)
            {
                throw new InvalidOperationException();
            }

            if (executor.GetData(HX).GradientAggregationCounter != 0)
            {
                throw new InvalidOperationException();
            }

            if (executor.GetData(CX).GradientAggregationCounter != 0)
            {
                throw new InvalidOperationException();
            }

            dnn.RNNBackwardData(
                executor.RnnDescDict[RnnDesc],
                descr.SeqLength,
                descr.YDesc,
                executor.GetTensor(Y).Buffer.Ptr,
                descr.YDesc,
                executor.GetGradient(Y).Buffer.Ptr,
                descr.StateDesc,               
                new deviceptr<T>(), // executor.GetGradient(HY).Buffer.Ptr,
                descr.StateDesc,              
                new deviceptr<T>(), // executor.GetGradient(CY).Buffer.Ptr,
                executor.FilterDescDict[WDesc],
                executor.GetTensor(W).Buffer.Ptr,
                descr.StateDesc,
                executor.GetTensor(HX).Buffer.Ptr,
                descr.StateDesc,
                executor.GetTensor(CX).Buffer.Ptr,
                descr.XDesc,
                executor.GetGradient(X).Buffer.Ptr,
                descr.StateDesc,
                executor.GetGradient(HX).Buffer.Ptr,
                descr.StateDesc,
                executor.GetGradient(CX).Buffer.Ptr,
                executor.GetTensor(Workspace).Buffer.Ptr,
                (IntPtr) executor.GetTensor(Workspace).Shape.Length,
                executor.GetTensor(ReserveSpace).Buffer.Ptr,
                (IntPtr) executor.GetTensor(ReserveSpace).Shape.Length);

            if (executor.GetData(W).GradientAggregationCounter == 0)
            {
                executor.AssignGradientDirectly(W, ScalarOps.Conv<T>(0.0).AsScalar());
            }

            dnn.RNNBackwardWeights(
                executor.RnnDescDict[RnnDesc],
                descr.SeqLength,
                descr.XDesc,
                executor.GetTensor(X).Buffer.Ptr,
                descr.StateDesc,
                executor.GetTensor(HX).Buffer.Ptr,
                descr.YDesc,
                executor.GetTensor(Y).Buffer.Ptr,
                executor.GetTensor(Workspace).Buffer.Ptr,
                (IntPtr) executor.GetTensor(Workspace).Shape.Length,
                executor.FilterDescDict[WDesc],
                executor.GetGradient(W).Buffer.Ptr,
                executor.GetTensor(ReserveSpace).Buffer.Ptr,
                (IntPtr) executor.GetTensor(ReserveSpace).Shape.Length);
        }
    }
}
