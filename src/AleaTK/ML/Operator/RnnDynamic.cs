using System;
using System.Linq;
using Alea;
using Alea.cuDNN;
using Alea.Parallel;
using static AleaTK.Library;
using static AleaTK.ML.Library;

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

        public TensorDescriptor StateDesc { get; }
        public TensorDescriptor[] XDesc { get; }
        public TensorDescriptor[] YDesc { get; }
        public Variable<T> W { get; }

        public readonly Variable<byte> DropoutStates = Library.AuxVariable<byte>();
        public readonly Variable<byte> Workspace = Library.AuxVariable<byte>();
        public readonly Variable<byte> ReserveSpace = Library.AuxVariable<byte>();
        public readonly Symbol DropoutDesc = new Symbol();
        public readonly Symbol WDesc = new Symbol();
        public readonly Symbol RnnDesc = new Symbol();

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
            StateDesc = new TensorDescriptor();
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

            var dropoutDesc = executor.DropoutDescDict[DropoutDesc];
            IntPtr dropoutStatesSize;
            dnn.DropoutGetStatesSize(out dropoutStatesSize);
            var dropoutStates = executor.GetTensor(DropoutStates, Shape.Create(dropoutStatesSize.ToInt64()));
            dropoutDesc.Set(dnn, (float) dropoutProbability, dropoutStates.Buffer.Ptr, dropoutStatesSize, dropoutSeed);

            var rnnDesc = executor.RnnDescDict[RnnDesc];
            var mode = rnnType.Mode;
            rnnDesc.Set(hiddenSize, numLayers, dropoutDesc, RNNInputMode.LINEAR_INPUT, DirectionMode.UNIDIRECTIONAL, mode, Dnn.DataTypeOf<T>());

            IntPtr workSize;
            dnn.GetRNNWorkspaceSize(rnnDesc, 1, XDesc, out workSize);
            executor.GetTensor(Workspace, Shape.Create(workSize.ToInt64()));

            if (isTraining)
            {
                IntPtr reserveSize;
                dnn.GetRNNTrainingReserveSize(rnnDesc, 1, XDesc, out reserveSize);
                executor.GetTensor(ReserveSpace, Shape.Create(reserveSize.ToInt64()));
            }

            var wDesc = executor.FilterDescDict[WDesc];
            IntPtr weightsSize;
            dnn.GetRNNParamsSize(rnnDesc, xDesc, out weightsSize, Dnn.DataTypeOf<T>());
            Util.EnsureTrue(weightsSize.ToInt64()%Gpu.SizeOf<T>() == 0);
            var shapeW = Shape.Create(weightsSize.ToInt64()/Alea.Gpu.SizeOf<T>());
            wDesc.SetND(Dnn.DataTypeOf<T>(), TensorFormat.CUDNN_TENSOR_NCHW, new[] {(int) shapeW[0], 1, 1});

            executor.GetTensor(W, shapeW);
            if (isTraining) executor.GetGradient(W, shapeW);
        }

        public void Initialize(Executor executor)
        {
            var context = executor.Context.ToGpuContext();
            var dnn = context.Dnn;

            var rnnDesc = executor.RnnDescDict[RnnDesc];
            var wDesc = executor.FilterDescDict[WDesc];

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
                        dnn.GetRNNLinLayerMatrixParams(rnnDesc, layer, XDesc[0], wDesc, w.Buffer.Ptr, linLayerId,
                            filterDesc, out linLayerMat);

                        filterDesc.GetND(out dataType, out format, out nbDims, filterDimA);
                        var length = filterDimA.Aggregate(ScalarOps.Mul);

                        var linLayerMatBuffer = new Buffer<T>(context.Device, w.Memory, new Layout(Shape.Create(length)),
                            linLayerMat);
                        var linLayerMatTensor = new Tensor<T>(linLayerMatBuffer);
                        context.Assign(linLayerMatTensor,
                            RandomNormal<T>(Shape.Create(length))/(Math.Sqrt(HiddenSize + InputSize).AsScalar<T>()));

                        deviceptr<T> linLayerBias;
                        dnn.GetRNNLinLayerBiasParams(rnnDesc, layer, XDesc[0], wDesc, w.Buffer.Ptr, linLayerId, filterDesc,
                            out linLayerBias);

                        filterDesc.GetND(out dataType, out format, out nbDims, filterDimA);
                        length = filterDimA.Aggregate(ScalarOps.Mul);

                        var linLayerBiasBuffer = new Buffer<T>(context.Device, w.Memory,
                            new Layout(Shape.Create(length)), linLayerBias);
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
            var rnnDesc = executor.RnnDescDict[RnnDesc];

            var hxDesc = StateDesc;
            var cxDesc = StateDesc;
            var hyDesc = StateDesc;
            var cyDesc = StateDesc;
            var wDesc = executor.FilterDescDict[WDesc];
            var w = executor.GetTensor(W);
            var workspace = executor.GetTensor(Workspace);

            if (IsTraining)
            {
                var reserveSpace = executor.GetTensor(ReserveSpace);
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
            var rnnDesc = executor.RnnDescDict[RnnDesc];
            var filterDesc = executor.FilterDescDict[WDesc];

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
                executor.GetTensor(Workspace).Buffer.Ptr,
                (IntPtr)executor.GetTensor(Workspace).Shape.Length,
                executor.GetTensor(ReserveSpace).Buffer.Ptr,
                (IntPtr)executor.GetTensor(ReserveSpace).Shape.Length);

            if (executor.GetData(W).GradientAggregationCounter == 0)
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
                executor.GetTensor(Workspace).Buffer.Ptr,
                (IntPtr)executor.GetTensor(Workspace).Shape.Length,
                executor.FilterDescDict[WDesc],
                executor.GetGradient(W).Buffer.Ptr,
                executor.GetTensor(ReserveSpace).Buffer.Ptr,
                (IntPtr)executor.GetTensor(ReserveSpace).Shape.Length);
        }
    }

    public class IteratedRnnCell<T> : Differentiable
    {
        public RnnType RnnType { get; }
        public bool IsTraining { get; }
        public int InputSize { get; }
        public int BatchSize { get; }
        public int HiddenSize { get; }
        public int NumLayers { get; }
        public double DropoutProbability { get; }
        public ulong DropoutSeed { get; }

        public Variable<T> Input { get; }
        public Variable<T> Output { get; }
        public Variable<T> W { get; }
        public Variable<T> H { get; }
        public Variable<T> C { get; }
        public Variable<T> DH { get; }
        public Variable<T> DC { get; }

        public readonly Symbol RnnCellDescr = new Symbol();

        public IteratedRnnCell(Variable<T> input, RnnType rnnRnnType, int hiddenSize, int numLayers, bool isTraining, double dropoutProbability, ulong dropoutSeed = 1337UL)
        {
            RnnType = rnnRnnType;
            IsTraining = isTraining;
            NumLayers = numLayers;
            HiddenSize = hiddenSize;
            DropoutProbability = isTraining ? dropoutProbability : 0.0;
            DropoutSeed = dropoutSeed;

            Util.EnsureEqual(3, input.Shape.Rank, "Input layout: (seqLength, batch, inputSize)");
            Util.EnsureTrue(input.Shape[1] >= 0, "Input layout: (seqLength, batch, inputSize)");
            Util.EnsureTrue(input.Shape[2] >= 0, "Input layout: (seqLength, batch, inputSize)");
            Input = input;
            BatchSize = (int)input.Shape[1];
            InputSize = (int)input.Shape[2];

            // output Shape (seqLength, batchSize, hiddenSize)
            Output = Variable<T>(PartialShape.Create(-1, BatchSize, HiddenSize));

            // W shape will be determined during initialization
            W = Parameter<T>();

            // state variables HX = H(0,:,:,:), HY = H(1,:,:,:)
            var shape = PartialShape.Create(2, NumLayers, BatchSize, HiddenSize);
            H = Variable<T>(shape);
            C = Variable<T>(shape);
            DH = Variable<T>(shape);
            DC = Variable<T>(shape);

            // construct the graph
            AddInput(Input);
            AddInput(W);
            AddOutput(Output);
            AddAuxVar(H);
            AddAuxVar(C);
            AddAuxVar(DH);
            AddAuxVar(DC);
        }

        public override void Initialize(Executor executor)
        {
            var cell = new RnnCell<T>(executor, RnnType, W, InputSize, BatchSize, HiddenSize, NumLayers, IsTraining, DropoutProbability, DropoutSeed);
            executor.Objects[RnnCellDescr] = cell;

            cell.Initialize(executor);

            base.Initialize(executor);
        }

        public void AssignInitialStates(Executor executor, Tensor<T> hx, Tensor<T> cx)
        {
            var shape = Shape.Create(2, NumLayers, BatchSize, HiddenSize);
            executor.Context.Assign(executor.GetTensor(H, shape).Slice(0), hx);
            executor.Context.Assign(executor.GetTensor(C, shape).Slice(0), cx);
        }

        public void AssignTerminalGradient(Executor executor, Tensor<T> dhy, Tensor<T> dcy)
        {
            var shape = Shape.Create(2, NumLayers, BatchSize, HiddenSize);
            executor.Context.Assign(executor.GetTensor(DH, shape).Slice(0), dhy);
            executor.Context.Assign(executor.GetTensor(DC, shape).Slice(0), dcy);
        }

        public void ZeroInitialStates(Executor executor)
        {
            executor.AssignTensor(H, Fill(Shape.Create(H.Shape.AsArray), ScalarOps.Conv<T>(0.0)));
            executor.AssignTensor(C, Fill(Shape.Create(C.Shape.AsArray), ScalarOps.Conv<T>(0.0)));
        }

        public void ZeroTerminalGradient(Executor executor)
        {
            executor.AssignTensor(DH, Fill(Shape.Create(H.Shape.AsArray), ScalarOps.Conv<T>(0.0)));
            executor.AssignTensor(DC, Fill(Shape.Create(C.Shape.AsArray), ScalarOps.Conv<T>(0.0)));
        }

        public override void Forward(Executor executor)
        {
            var cell = (RnnCell<T>) executor.Objects[RnnCellDescr];

            // input should be allocated by the training procedure
            var input = executor.GetTensor(Input);
            var seqLength = input.Shape[0];

            // output is the output of this op, so we need give shape to allocate it
            var output = executor.GetTensor(Output, Shape.Create(seqLength, BatchSize, HiddenSize));

            var shape = Shape.Create(2, NumLayers, BatchSize, HiddenSize);
            var h = executor.GetTensor(H, shape);
            var c = executor.GetTensor(C, shape);

            for (var t = 0; t < seqLength; ++t)
            {
                cell.Input = input.Slice(t);
                cell.Output = output.Slice(t);

                var i = t % 2;
                cell.HX = h.Slice(i).Reshape(NumLayers, BatchSize, HiddenSize);
                cell.HY = h.Slice(1 - i).Reshape(NumLayers, BatchSize, HiddenSize);
                cell.CX = c.Slice(i).Reshape(NumLayers, BatchSize, HiddenSize);
                cell.CY = c.Slice(1 - i).Reshape(NumLayers, BatchSize, HiddenSize);

                cell.Forward(executor);
            }
        }

        public override void Backward(Executor executor)
        {
            var cell = (RnnCell<T>)executor.Objects[RnnCellDescr];

            // input and output should be there, cause this is backward()
            var input = executor.GetTensor(Input);
            var output = executor.GetTensor(Output);

            // dOutput should be allocated by the child op, but dInput should be allocated by us
            var dInput = executor.GetGradient(Input, input.Shape);
            var dOutput = executor.GetGradient(Output);

            var shape = Shape.Create(2, NumLayers, BatchSize, HiddenSize);
            var h = executor.GetTensor(H, shape);
            var c = executor.GetTensor(C, shape);
            var dh = executor.GetTensor(DH, shape);
            var dc = executor.GetTensor(DC, shape);

            var seqLength = (int)Input.Shape[0];

            for (var t = seqLength - 1; t >= 0; --t)
            {
                cell.Input = input.Slice(t);
                cell.Output = output.Slice(t);
                cell.DOutput = dOutput.Slice(t);
                cell.DInput = dInput.Slice(t);

                var i = (seqLength - 1 - t) % 2;
                cell.HX = h.Slice(i).Reshape(NumLayers, BatchSize, HiddenSize);
                cell.HY = h.Slice(1 - i).Reshape(NumLayers, BatchSize, HiddenSize);
                cell.CX = c.Slice(i).Reshape(NumLayers, BatchSize, HiddenSize);
                cell.CY = c.Slice(1 - i).Reshape(NumLayers, BatchSize, HiddenSize);
                cell.DHX = dh.Slice(i).Reshape(NumLayers, BatchSize, HiddenSize);
                cell.DHY = dh.Slice(1 - i).Reshape(NumLayers, BatchSize, HiddenSize);
                cell.DCX = dc.Slice(i).Reshape(NumLayers, BatchSize, HiddenSize);
                cell.DCY = dc.Slice(1 - i).Reshape(NumLayers, BatchSize, HiddenSize);

                cell.Backward(executor);
            }
        }
    }

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

                executor.GetGradient(Rnn.X, x.Shape);
                executor.GetGradient(Rnn.Y, Shape.Create(SeqLength, MiniBatch, Rnn.HiddenSize));
                executor.GetGradient(Rnn.HX, Shape.Create(Rnn.NumLayers, MiniBatch, Rnn.HiddenSize));
                executor.GetGradient(Rnn.CX, Shape.Create(Rnn.NumLayers, MiniBatch, Rnn.HiddenSize));
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

        public void ZeroInitialStates(Executor executor)
        {
            executor.AssignTensor(Rnn.HX, Fill(Shape.Create(Rnn.HX.Shape.AsArray), ScalarOps.Conv<T>(0.0)));
            executor.AssignTensor(Rnn.CX, Fill(Shape.Create(Rnn.CX.Shape.AsArray), ScalarOps.Conv<T>(0.0)));
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
        public RnnDynamic(RnnType rnnRnnType, Variable<T> x, int numLayers, int hiddenSize, bool isTraining = true, double dropout = 0.0, ulong dropoutSeed = 1337UL)
        {
            RnnType = rnnRnnType;
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

        public RnnType RnnType { get; }

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
            var mode = RnnType.Mode;
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
            var shapeW = Shape.Create(weightsSize.ToInt64() / Alea.Gpu.SizeOf<T>());
            wDesc.SetND(Dnn.DataTypeOf<T>(), TensorFormat.CUDNN_TENSOR_NCHW, new [] {(int)shapeW[0], 1, 1});

            // since we are using cuDNN, we'd better make sure these varaibles are allocated
            executor.GetTensor(W, shapeW);
            if (IsTraining) executor.GetGradient(W, shapeW);

            // init weights
            var numLinearLayers = RnnType.NumLinLayers;

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
                        RnnType.InitBias(context, layer, linLayerId, linLayerBiasTensor);
                    }
                }
            }

            base.Initialize(executor);
        }

        /// <summary>
        /// Call ZeroInitialStates at least once before Forward or Backward. 
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
        /// Call ZeroInitialStates at least once before Forward or Backward. 
        /// </summary>
        /// <param name="executor"></param>
        public void ZeroInitialStates(Executor executor)
        {
            var descr = (RnnDynamicDescr<T>)executor.Objects[Descr];
            descr.ZeroInitialStates(executor);
        }

        public override void Forward(Executor executor)
        {
            var descr = new RnnDynamicDescr<T>(executor, this);
            executor.Objects[Descr] = descr;

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
