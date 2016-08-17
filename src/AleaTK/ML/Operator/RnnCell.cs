using System;
using System.Linq;
using Alea;
using Alea.cuDNN;

namespace AleaTK.ML.Operator
{
    public class RnnCell<T>
    {
        /// <summary>
        /// Input tensor of size [batch, inputDim]
        /// </summary>
        public Tensor<T> Input { get; set; }
        public Tensor<T> DInput { get; set; }
        public Tensor<T> Output { get; set; }
        public Tensor<T> DOutput { get; set; }

        /// <summary>
        /// HX, CX, HY, CY tensors of size [numLayers, batch, hiddenSize]
        /// </summary>
        public Tensor<T> HX { get; set; }
        public Tensor<T> DHX { get; set; }
        public Tensor<T> CX { get; set; }
        public Tensor<T> DCX { get; set; }
        public Tensor<T> HY { get; set; }
        public Tensor<T> DHY { get; set; }
        public Tensor<T> CY { get; set; }
        public Tensor<T> DCY { get; set; }

        public TensorDescriptor StateDesc { get; } = new TensorDescriptor();
        public TensorDescriptor[] XDesc { get; }
        public TensorDescriptor[] YDesc { get; }
        public Variable<T> W { get; }

        public Tensor<byte> DropoutStates { get; }
        public Tensor<byte> Workspace { get; }
        public Tensor<byte> ReserveSpace { get; set; }
        public long ReserveSize { get; }

        public DropoutDescriptor DropoutDesc { get; } = new DropoutDescriptor();
        public FilterDescriptor WDesc { get; } = new FilterDescriptor();
        public RNNDescriptor RnnDesc { get; } = new RNNDescriptor();

        public bool IsTraining { get; }
        public int BatchSize { get; }
        public int InputSize { get; }
        public int HiddenSize { get; }
        public int NumLayers { get; }
        public RnnType RnnType { get; }

        public RnnCell(Executor executor, RnnType rnnType, Variable<T> w, int inputSize, int batch, int hiddenSize,
            int numLayers, bool isTraining, double dropoutProbability, ulong dropoutSeed = 1337UL)
        {
            IsTraining = isTraining;
            BatchSize = batch;
            InputSize = inputSize;
            HiddenSize = hiddenSize;
            NumLayers = numLayers;
            RnnType = rnnType;
            W = w;

            var context = executor.Context.ToGpuContext();
            var dnn = context.Dnn;

            // state variables
            var shape = Shape.Create(numLayers, batch, hiddenSize);
            var strides = Strides.Create(shape[1]*shape[2], shape[2], 1); // inner change most
            StateDesc.SetND(Dnn.DataTypeOf<T>(), shape.AsInt32Array, strides.AsInt32Array);

            // xDesc is an array of one element because we do only one step
            shape = Shape.Create(batch, inputSize, 1);
            strides = Strides.Create(shape[1]*shape[2], shape[2], 1);
            var xDesc = new TensorDescriptor();
            xDesc.SetND(Dnn.DataTypeOf<T>(), shape.AsInt32Array, strides.AsInt32Array);
            XDesc = Enumerable.Repeat(xDesc, 1).ToArray();

            // yDesc is an array of one element because we do only one step
            shape = Shape.Create(batch, hiddenSize, 1);
            strides = Strides.Create(shape[1]*shape[2], shape[2], 1);
            var yDesc = new TensorDescriptor();
            yDesc.SetND(Dnn.DataTypeOf<T>(), shape.AsInt32Array, strides.AsInt32Array);
            YDesc = Enumerable.Repeat(yDesc, 1).ToArray();

            IntPtr dropoutStatesSize;
            dnn.DropoutGetStatesSize(out dropoutStatesSize);
            DropoutStates = executor.Context.Device.Allocate<byte>(Shape.Create(dropoutStatesSize.ToInt64()));
            DropoutDesc.Set(dnn, (float) dropoutProbability, DropoutStates.Buffer.Ptr, dropoutStatesSize, dropoutSeed);

            var mode = rnnType.Mode;
            RnnDesc.Set(hiddenSize, numLayers, DropoutDesc, RNNInputMode.LINEAR_INPUT, DirectionMode.UNIDIRECTIONAL, mode, Dnn.DataTypeOf<T>());

            IntPtr workSize;
            dnn.GetRNNWorkspaceSize(RnnDesc, 1, XDesc, out workSize);
            Workspace = executor.Context.Device.Allocate<byte>(Shape.Create(workSize.ToInt64()));

            if (isTraining)
            {
                IntPtr reserveSize;
                dnn.GetRNNTrainingReserveSize(RnnDesc, 1, XDesc, out reserveSize);
                ReserveSize = reserveSize.ToInt64();
                //ReserveSpace = executor.AttentionState.Device.Allocate<byte>(Shape.Create(reserveSize.ToInt64()));
            }

            IntPtr weightsSize;
            dnn.GetRNNParamsSize(RnnDesc, xDesc, out weightsSize, Dnn.DataTypeOf<T>());
            Util.EnsureTrue(weightsSize.ToInt64()%Gpu.SizeOf<T>() == 0);
            var shapeW = Shape.Create(weightsSize.ToInt64()/Alea.Gpu.SizeOf<T>());
            WDesc.SetND(Dnn.DataTypeOf<T>(), TensorFormat.CUDNN_TENSOR_NCHW, new[] {(int) shapeW[0], 1, 1});

            executor.GetTensor(W, shapeW);
            if (isTraining) executor.GetGradient(W, shapeW);
        }

        public void Initialize(Executor executor)
        {
            var context = executor.Context.ToGpuContext();
            var dnn = context.Dnn;

            var rnnDesc = RnnDesc;
            var wDesc = WDesc;

            // init weights
            using (var filterDesc = new FilterDescriptor())
            {
                var w = executor.GetTensor(W);
                var filterDimA = new int[3];

                for (var layer = 0; layer < NumLayers; ++layer)
                {
                    for (var linLayerId = 0; linLayerId < RnnType.NumLinLayers; ++linLayerId)
                    {
                        int nbDims;
                        DataType dataType;
                        TensorFormat format;

                        deviceptr<T> linLayerMat;
                        dnn.GetRNNLinLayerMatrixParams(rnnDesc, layer, XDesc[0], wDesc, w.Buffer.Ptr, linLayerId, filterDesc, out linLayerMat);

                        filterDesc.GetND(out dataType, out format, out nbDims, filterDimA);
                        var length = filterDimA.Aggregate(ScalarOps.Mul);

                        var linLayerMatBuffer = new Buffer<T>(context.Device, w.Memory, new Layout(Shape.Create(length)), linLayerMat);
                        var linLayerMatTensor = new Tensor<T>(linLayerMatBuffer);
                        context.Assign(linLayerMatTensor, AleaTK.Library.RandomNormal<T>(Shape.Create(length))/(Math.Sqrt(HiddenSize + InputSize).AsScalar<T>()));

                        deviceptr<T> linLayerBias;
                        dnn.GetRNNLinLayerBiasParams(rnnDesc, layer, XDesc[0], wDesc, w.Buffer.Ptr, linLayerId, filterDesc, out linLayerBias);

                        filterDesc.GetND(out dataType, out format, out nbDims, filterDimA);
                        length = filterDimA.Aggregate(ScalarOps.Mul);

                        var linLayerBiasBuffer = new Buffer<T>(context.Device, w.Memory, new Layout(Shape.Create(length)), linLayerBias);
                        var linLayerBiasTensor = new Tensor<T>(linLayerBiasBuffer);
                        RnnType.InitBias(context, layer, linLayerId, linLayerBiasTensor);
                    }
                }
            }
        }

        public void Forward(Executor executor)
        {
            var context = executor.Context.ToGpuContext();
            var dnn = context.Dnn;
            var rnnDesc = RnnDesc;

            var hxDesc = StateDesc;
            var cxDesc = StateDesc;
            var hyDesc = StateDesc;
            var cyDesc = StateDesc;
            var wDesc = WDesc;
            var w = executor.GetTensor(W);
            var workspace = Workspace;

            if (IsTraining)
            {
                var reserveSpace = ReserveSpace;
                dnn.RNNForwardTraining(
                    rnnDesc, 1, XDesc, Input.Buffer.Ptr, hxDesc, HX.Buffer.Ptr,
                    cxDesc, CX.Buffer.Ptr, wDesc, w.Buffer.Ptr, YDesc, Output.Buffer.Ptr,
                    hyDesc, HY.Buffer.Ptr, cyDesc, CY.Buffer.Ptr,
                    workspace.Buffer.Ptr, (IntPtr)workspace.Shape.Length,
                    reserveSpace.Buffer.Ptr, (IntPtr)reserveSpace.Shape.Length);
            }
            else
            {
                dnn.RNNForwardInference(
                    rnnDesc, 1, XDesc, Input.Buffer.Ptr, hxDesc, HX.Buffer.Ptr,
                    cxDesc, CX.Buffer.Ptr, wDesc, w.Buffer.Ptr, YDesc, Output.Buffer.Ptr,
                    hyDesc, HY.Buffer.Ptr, cyDesc, CY.Buffer.Ptr,
                    workspace.Buffer.Ptr, (IntPtr)workspace.Shape.Length);
            }
        }

        public void Backward(Executor executor)
        {
            var context = executor.Context.ToGpuContext();
            var dnn = context.Dnn;
            var rnnDesc = RnnDesc;
            var filterDesc = WDesc;

            Util.EnsureTrue(IsTraining);

            dnn.RNNBackwardData(
                rnnDesc,
                1,
                YDesc,
                Output.Buffer.Ptr,
                YDesc,
                DOutput.Buffer.Ptr,
                StateDesc,
                DHY.Buffer.Ptr,
                StateDesc,
                DCY.Buffer.Ptr,
                filterDesc,
                executor.GetTensor(W).Buffer.Ptr,
                StateDesc,
                HX.Buffer.Ptr,
                StateDesc,
                CX.Buffer.Ptr,
                XDesc,
                DInput.Buffer.Ptr,
                StateDesc,
                DHX.Buffer.Ptr,
                StateDesc,
                DCX.Buffer.Ptr,
                Workspace.Buffer.Ptr,
                (IntPtr)Workspace.Shape.Length,
                ReserveSpace.Buffer.Ptr,
                (IntPtr)ReserveSpace.Shape.Length);

            if (executor.GetData(W).CheckGradientAggregationCounter == 0)
            {
                executor.AssignGradientDirectly(W, ScalarOps.Conv<T>(0.0).AsScalar());
            }

            dnn.RNNBackwardWeights(
                rnnDesc,
                1,
                XDesc,
                Input.Buffer.Ptr,
                StateDesc,
                HX.Buffer.Ptr,
                YDesc,
                Output.Buffer.Ptr,
                Workspace.Buffer.Ptr,
                (IntPtr)Workspace.Shape.Length,
                WDesc,
                executor.GetGradient(W).Buffer.Ptr,
                ReserveSpace.Buffer.Ptr,
                (IntPtr)ReserveSpace.Shape.Length);
        }
    }
}