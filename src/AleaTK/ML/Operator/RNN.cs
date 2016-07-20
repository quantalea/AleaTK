using System;
using System.Linq;
using Alea;
using Alea.cuDNN;
using static AleaTK.Library;
using static AleaTK.ML.Library;

namespace AleaTK.ML.Operator
{

    public class LSTM<T> : Differentiable
    {
        public LSTM(Variable<T> x, int hiddenSize, Variable<T> cx = null, Variable<T> hx = null, double forgetBiasInit = 3.0)
        {
            // X shape (seqLength, batch, inputSize)
            Util.EnsureEqual(3, x.Shape.Rank, "Input layout: (seqLength, batch, inputSize)");
            X = x;
            SeqLength = (int)X.Shape[0];
            InputSize = (int)X.Shape[2];
            HiddenSize = hiddenSize;
            ForgetBiasInit = forgetBiasInit;

            // Y Shape (seqLength, batch, hiddenSize)
            Y = Variable<T>(PartialShape.Create(SeqLength, -1, HiddenSize));

            // W (inputSize + hiddenSize + 1, 4 * hiddenSize)
            W =
                Parameter(RandomNormal<T>(Shape.Create(InputSize + HiddenSize + 1, 4*HiddenSize))/
                          (Math.Sqrt(InputSize + hiddenSize)).AsScalar<T>());
            // the following W initialization happens in Initialize();

            Hin = AuxVariable<T>();
            Hout = AuxVariable<T>();
            IFOG = AuxVariable<T>();
            IFOGf = AuxVariable<T>();
            C = AuxVariable<T>();
            Ct = AuxVariable<T>();
            Temp1 = AuxVariable<T>();
            CX = cx ?? Variable<T>();
            HX = hx ?? Variable<T>();

            AddInput(X);
            AddOutput(Y);
            AddInput(W);
            AddInput(CX);
            AddInput(HX);
            AddAuxVar(Hin);
            AddAuxVar(Hout);
            AddAuxVar(IFOG);
            AddAuxVar(IFOGf);
            AddAuxVar(C);
            AddAuxVar(Ct);
            AddAuxVar(Temp1);
        }

        public override void Initialize(Executor executor)
        {
            base.Initialize(executor);

            // set bias to zero
            var ctx = executor.Context;
            var w = executor.GetTensor(W);
            ctx.Assign(w.Slice(0), 0.0.AsScalar<T>());

            if (ForgetBiasInit != 0.0)
            {
                ctx.Assign(w.Slice(0, Range(HiddenSize, HiddenSize*2)),
                    Fill(Shape.Create(1, HiddenSize), ScalarOps.Conv<T>(ForgetBiasInit)));
            }
        }

        public double ForgetBiasInit { get; }

        public int SeqLength { get; }

        public int InputSize { get; }

        public int HiddenSize { get; }

        public Variable<T> X { get; }

        public Variable<T> Y { get; }

        public Variable<T> W { get; }

        public Variable<T> CX { get; }

        public Variable<T> HX { get; }

        public Variable<T> Hin { get; }

        public Variable<T> Hout { get; }

        public Variable<T> IFOG { get; }

        public Variable<T> IFOGf { get; }

        public Variable<T> C { get; }

        public Variable<T> Ct { get; }

        public Variable<T> Temp1 { get; }

        public override void Forward(Executor executor)
        {
            var w = executor.GetTensor(W);
            var xphpb = w.Shape[0];
            var x = executor.GetTensor(X);
            var b = x.Shape[1];
            var n = x.Shape[0];
            var d = HiddenSize;

            var c0 = executor.GetTensor(CX);
            var h0 = executor.GetTensor(HX);
            Util.EnsureTrue(c0.Shape.SequenceEqual(Shape.Create(b, d)));
            Util.EnsureTrue(h0.Shape.SequenceEqual(Shape.Create(b, d)));

            var hin = executor.GetTensor(Hin, Shape.Create(n, b, xphpb));
            var hout = executor.GetTensor(Hout, Shape.Create(n, b, d));
            var ifog = executor.GetTensor(IFOG, Shape.Create(n, b, d*4));
            var ifogf = executor.GetTensor(IFOGf, Shape.Create(n, b, d*4));
            var c = executor.GetTensor(C, Shape.Create(n, b, d));
            var ct = executor.GetTensor(Ct, Shape.Create(n, b, d));

            var ctx = executor.Context;

            for (var t = 0; t < n; ++t)
            {
<<<<<<< 48ffa75a6d00ddbc17f3e7bf64bfec7edd6ceff2
                ctx.Assign(prevh, t > 0 ? hout.Slice(Range.Create(t - 1), Range.All, Range.All) : 0.0.AsScalar<T>());

                ctx.Assign(hin.Slice(Range.Create(t), Range.All, Range.Create(0)), Fill(Shape.Create(1, b, 1), ScalarOps.Conv<T>(1.0))); // bias
                ctx.Assign(hin.Slice(Range.Create(t), Range.All, Range.Create(1, InputSize + 1)),
                    x.Slice(Range.Create(t), Range.All, Range.All));
                ctx.Assign(hin.Slice(Range.Create(t), Range.All, Range.Create(InputSize + 1, -1)), prevh);

                //Console.WriteLine(hin.Shape);
                //Console.WriteLine(w.Shape);
                ctx.Assign(ifog.Slice(Range.Create(t), Range.All, Range.All),
                    Dot(hin.Slice(Range.Create(t), Range.All, Range.All).Reshape(b, xphpb), w));

                ctx.Assign(ifogf.Slice(Range.Create(t), Range.All, Range.Create(0, 3 * d)),
                    1.0.AsScalar<T>() /
                    (1.0.AsScalar<T>() + Exp(-ifog.Slice(Range.Create(t), Range.All, Range.Create(0, 3 * d)))));

                ctx.Assign(ifogf.Slice(Range.Create(t), Range.All, Range.Create(3 * d, -1)),
                    Tanh(ifog.Slice(Range.Create(t), Range.All, Range.Create(3 * d, -1))));

                ctx.Assign(prevc, t > 0 ? c.Slice(Range.Create(t - 1), Range.All, Range.All) : 0.0.AsScalar<T>());

                ctx.Assign(c.Slice(Range.Create(t), Range.All, Range.All),
                    ifogf.Slice(Range.Create(t), Range.All, Range.Create(0, d)) *
                    ifogf.Slice(Range.Create(t), Range.All, Range.Create(3 * d, -1)) +
                    ifogf.Slice(Range.Create(t), Range.All, Range.Create(d, 2 * d)) * prevc);

                ctx.Assign(ct.Slice(Range.Create(t), Range.All, Range.All),
                    Tanh(c.Slice(Range.Create(t), Range.All, Range.All)));

                ctx.Assign(hout.Slice(Range.Create(t), Range.All, Range.All),
                    ifogf.Slice(Range.Create(t), Range.All, Range.Create(2 * d, 3 * d)) *
                    ct.Slice(Range.Create(t), Range.All, Range.All));
=======
                // stack input
                var prevh = executor.GetTensor(Temp1, Shape.Create(1, b, d));
                ctx.Assign(prevh, t > 0 ? hout.Slice(t - 1) : h0);
                ctx.Assign(hin.Slice(t, -1, 0), Fill(Shape.Create(1, b, 1), ScalarOps.Conv<T>(1.0))); // bias
                ctx.Assign(hin.Slice(t, -1, Range(1, InputSize + 1)), x.Slice(t));
                ctx.Assign(hin.Slice(t, -1, Range(InputSize + 1, -1)), prevh);

                // dot
                ctx.Assign(ifog.Slice(t), Dot(hin.Slice(t).Reshape(b, xphpb), w));

                // non-linearities
                // first 3 matrices are ifo
                ctx.Assign(ifogf.Slice(t, -1, Range(0, 3*d)), 
                    1.0.AsScalar<T>() / (1.0.AsScalar<T>() + Exp(-ifog.Slice(t, -1, Range(0, 3*d)))));

                // last one is for g(a)
                ctx.Assign(ifogf.Slice(t, -1, Range(3*d, -1)), Tanh(ifog.Slice(t, -1, Range(3*d, -1))));

                // update c
                var prevc = executor.GetTensor(Temp1, Shape.Create(1, b, d));
                ctx.Assign(prevc, t > 0 ? c.Slice(t - 1) : c0);
                // c_t = i_t * a_t + f_t * c_t-1
                ctx.Assign(c.Slice(t),
                    ifogf.Slice(t, -1, Range(0, d)) * ifogf.Slice(t, -1, Range(3 * d, -1)) +
                    ifogf.Slice(t, -1, Range(d, 2 * d)) * prevc);
                // h_t = o_t * tanh(c_t)
                ctx.Assign(ct.Slice(t), Tanh(c.Slice(t)));
                ctx.Assign(hout.Slice(t), ifogf.Slice(t, -1, Range(2 * d, 3 * d)) * ct.Slice(t));
            }

            executor.AssignTensor(Y, hout);
        }

        public static Tensor<T> GetZeroGradient(Executor executor, Variable<T> var)
        {
            var data = executor.GetData(var);
            Util.EnsureTrue(data.GradientAggregationCounter == 0);
            var tensor = executor.GetTensor(var);
            executor.AssignGradientDirectly(var, Fill(tensor.Shape, ScalarOps.Conv<T>(0.0)));
            return executor.GetGradient(var);
        }

        public override void Backward(Executor executor)
        {
            var ctx = executor.Context;

            var dy = executor.GetGradient(Y); // input
            var w = executor.GetTensor(W);
            var x = executor.GetTensor(X);
            var c = executor.GetTensor(C);
            var ct = executor.GetTensor(Ct);
            var hin = executor.GetTensor(Hin);
            var hout = executor.GetTensor(Hout);
            var ifogf = executor.GetTensor(IFOGf);
            var n = hout.Shape[0];
            var b = hout.Shape[1];
            var d = (int)hout.Shape[2];

            var c0 = executor.GetTensor(CX);
            var h0 = executor.GetTensor(HX);
            Util.EnsureTrue(c0.Shape.SequenceEqual(Shape.Create(b, d)));
            Util.EnsureTrue(h0.Shape.SequenceEqual(Shape.Create(b, d)));

            var dc = GetZeroGradient(executor, C);
            var dx = GetZeroGradient(executor, X);
            var dw = GetZeroGradient(executor, W);
            var dIFOG = GetZeroGradient(executor, IFOG);
            var dIFOGf = GetZeroGradient(executor, IFOGf);
            var dhin = GetZeroGradient(executor, Hin);
            var dhout = GetZeroGradient(executor, Hout);
            var dh0 = GetZeroGradient(executor, HX);
            var dc0 = GetZeroGradient(executor, CX);

            ctx.Assign(dhout, dy);

            // TODO: dcn and dhn
            // now all are 0!

            for (var t = n - 1; t >= 0; --t)
            {
                var tanhCt = ct.Slice(t);

                // do_t = dh_t * tanh(c_t)
                ctx.Assign(dIFOGf.Slice(t, -1, Range(2*d, 3*d)), tanhCt*dhout.Slice(t));

                // dc_t += dh_t * o_t * (1 - tanh**2(c_t))
                ctx.Assign(dc.Slice(t),
                    dc.Slice(t) +
                    (1.0.AsScalar<T>() - tanhCt*tanhCt)*(ifogf.Slice(t, -1, Range(2*d, 3*d))*dhout.Slice(t)));

                // df_t = dc_t * c_t-1
                if (t > 0)
                {
                    ctx.Assign(dIFOGf.Slice(t, -1, Range(d, 2*d)), c.Slice(t - 1)*dc.Slice(t));
                    ctx.Assign(dc.Slice(t - 1), dc.Slice(t - 1) + ifogf.Slice(t, -1, Range(d, 2*d))*dc.Slice(t));
                }
                else
                {
                    ctx.Assign(dIFOGf.Slice(t, -1, Range(d, 2*d)), c0*dc.Slice(t));
                    ctx.Assign(dc0, (ifogf.Slice(t, -1, Range(d, 2*d))*dc.Slice(t)).Reshape(b, d));
                }
                // di_t = dc_t * a_t
                ctx.Assign(dIFOGf.Slice(t, -1, Range(0, d)), ifogf.Slice(t, -1, Range(3*d, -1))*dc.Slice(t));
                // da_t = dc_t * i_t
                ctx.Assign(dIFOGf.Slice(t, -1, Range(3*d, -1)), ifogf.Slice(t, -1, Range(0, d))*dc.Slice(t));

                // backprop activation functions
                var tmp1 = ifogf.Slice(t, -1, Range(3*d, -1));
                ctx.Assign(dIFOG.Slice(t, -1, Range(3*d, -1)), (1.0.AsScalar<T>() - tmp1 * tmp1)*dIFOGf.Slice(t, -1, Range(3*d, -1)));
                var tmp2 = ifogf.Slice(t, -1, Range(0, 3*d));
                ctx.Assign(dIFOG.Slice(t, -1, Range(0, 3*d)),
                    (tmp2*(1.0.AsScalar<T>() - tmp2))*dIFOGf.Slice(t, -1, Range(0, 3*d)));

                // backprop matrix multiply
                ctx.Assign(dw, dw + Dot(hin.Slice(t).T, dIFOG.Slice(t)));
                ctx.Assign(dhin.Slice(t), Dot(dIFOG.Slice(t), w.T));

                // backprop the identity transforms into hin
                ctx.Assign(dx.Slice(t), dhin.Slice(t, -1, Range(1, InputSize + 1)));
                if (t > 0)
                {
                    ctx.Assign(dhout.Slice(t - 1), dhout.Slice(t - 1) + dhin.Slice(t, -1, Range(InputSize + 1, -1)));
                }
                else
                {
                    ctx.Assign(dh0, dh0 + dhin.Slice(t, -1, Range(InputSize + 1, -1)));
                }
>>>>>>> 89707ada1ea7cf18d1d9ec732073f0aec84dd67d
            }
        }
    }

    public class RNN<T> : Differentiable
    {
        public RNN(Variable<T> x, int numLayers, int hiddenSize, bool isTraining = true, double dropout = 0.0, double bias = 0.0, ulong dropoutSeed = 1337UL)
        {
            X = x;
            IsTraining = isTraining;
            NumLayers = numLayers;
            HiddenSize = hiddenSize;
            Bias = bias;
            Dropout = dropout;
            DropoutSeed = dropoutSeed;

            // X shape (batch, seqLength, inputSize)
            Util.EnsureEqual(3, X.Shape.Rank, "Input layout: (batch, seqLength, inputSize)");
            MiniBatch = (int) X.Shape[0];
            SeqLength = (int) X.Shape[1];
            InputSize = (int) X.Shape[2];

            // Y Shape (batch, seqLength, hiddenSize)
            Y = Library.Variable<T>(PartialShape.Create(MiniBatch, SeqLength, HiddenSize));

            // W shape will be determined during initialization
            W = Library.Parameter<T>();

            // state variables
            var shape = PartialShape.Create(NumLayers, MiniBatch, HiddenSize);
            var strides = Strides.Create(shape[1]*shape[2], shape[2], 1); // inner change most
            HX = Library.Variable<T>(shape);
            CX = Library.Variable<T>(shape);
            //HX = Parameter(Fill(Shape.Create(NumLayers, MiniBatch, HiddenSize), ScalarOps.Conv<T>(0.0)));
            //CX = Parameter(Fill(Shape.Create(NumLayers, MiniBatch, HiddenSize), ScalarOps.Conv<T>(0.0)));
            HY = Library.Variable<T>(shape);
            CY = Library.Variable<T>(shape);
            StateDesc = new TensorDescriptor();
            StateDesc.SetND(Dnn.DataTypeOf<T>(), shape.AsInt32Array, strides.AsInt32Array);

            // xDesc is an array, for each step
            shape = PartialShape.Create(MiniBatch, InputSize, 1);
            strides = Strides.Create(shape[1]*shape[2], shape[2], 1);
            var xDesc = new TensorDescriptor();
            xDesc.SetND(Dnn.DataTypeOf<T>(), shape.AsInt32Array, strides.AsInt32Array);
            XDesc = Enumerable.Repeat(xDesc, SeqLength).ToArray();

            // yDesc is an array, for each step
            shape = PartialShape.Create(MiniBatch, HiddenSize, 1);
            strides = Strides.Create(shape[1]*shape[2], shape[2], 1);
            var yDesc = new TensorDescriptor();
            yDesc.SetND(Dnn.DataTypeOf<T>(), shape.AsInt32Array, strides.AsInt32Array);
            YDesc = Enumerable.Repeat(yDesc, SeqLength).ToArray();

            // construct the graph
            AddInput(X);
            AddInput(W);
            AddOutput(Y);
            AddAuxVar(HX);
            AddAuxVar(CX);
            //AddInput(HX);
            //AddInput(CX);
            AddAuxVar(HY);
            AddAuxVar(CY);
            AddAuxVar(DropoutStates);
            AddAuxVar(Workspace);
            AddAuxVar(ReserveSpace);
        }

        public bool IsTraining { get; }

        public double Dropout { get; }

        public double Bias { get; }

        public ulong DropoutSeed { get; }

        public TensorDescriptor StateDesc { get; }

        public TensorDescriptor[] XDesc { get; }

        public TensorDescriptor[] YDesc { get; }

        public int NumLayers { get; }

        public int HiddenSize { get; }

        public int SeqLength { get; }

        public int MiniBatch { get; }

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

        public void InitializeStates(Executor executor)
        {
            const double value = 0.0;

            executor.AssignTensor(HX, AleaTK.Library.Fill(Shape.Create(HX.Shape.AsArray), ScalarOps.Conv<T>(value)));
            executor.AssignTensor(CX, AleaTK.Library.Fill(Shape.Create(CX.Shape.AsArray), ScalarOps.Conv<T>(value)));
            executor.AssignTensor(HY, AleaTK.Library.Fill(Shape.Create(HY.Shape.AsArray), ScalarOps.Conv<T>(value)));
            executor.AssignTensor(CY, AleaTK.Library.Fill(Shape.Create(HY.Shape.AsArray), ScalarOps.Conv<T>(value)));

            // we assign them directly (no gradient counter increasing)
            if (IsTraining)
            {
                executor.AssignGradientDirectly(HY, AleaTK.Library.Fill(Shape.Create(HY.Shape.AsArray), ScalarOps.Conv<T>(value)));
                executor.AssignGradientDirectly(CY, AleaTK.Library.Fill(Shape.Create(CY.Shape.AsArray), ScalarOps.Conv<T>(value)));
                executor.AssignGradientDirectly(W, ScalarOps.Conv<T>(0.0).AsScalar());
            }
        }

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
            var mode = RNNMode.LSTM;
            rnnDesc.Set(HiddenSize, NumLayers, dropoutDesc, RNNInputMode.LINEAR_INPUT, DirectionMode.UNIDIRECTIONAL,
                mode, Dnn.DataTypeOf<T>());

            // weight
            var wDesc = executor.FilterDescDict[WDesc];
            IntPtr weightsSize;
            dnn.GetRNNParamsSize(rnnDesc, XDesc[0], out weightsSize, Dnn.DataTypeOf<T>());
            Util.EnsureTrue(weightsSize.ToInt64()%Gpu.SizeOf<T>() == 0);
            var shapeW = Shape.Create(weightsSize.ToInt64()/Alea.Gpu.SizeOf<T>(), 1, 1);
            wDesc.SetND(Dnn.DataTypeOf<T>(), TensorFormat.CUDNN_TENSOR_NCHW, shapeW.AsInt32Array);
            //Console.WriteLine($"RNN.W.Shape: {shapeW}");

            // workspace and rreservespace
            IntPtr workSize;
            dnn.GetRNNWorkspaceSize(rnnDesc, SeqLength, XDesc, out workSize);
            executor.GetTensor(Workspace, Shape.Create(workSize.ToInt64()));

            if (IsTraining)
            {
                IntPtr reserveSize;
                dnn.GetRNNTrainingReserveSize(rnnDesc, SeqLength, XDesc, out reserveSize);
                executor.GetTensor(ReserveSpace, Shape.Create(reserveSize.ToInt64()));
            }

            // since we are using cuDNN, we'd better make sure these varaibles are allocated
            executor.GetTensor(W, shapeW);
            if (IsTraining) executor.GetGradient(W, shapeW);
            
            executor.GetTensor(Y, (Shape.Create(Y.Shape.AsArray)));
            executor.GetTensor(HX, (Shape.Create(HX.Shape.AsArray)));
            executor.GetTensor(CX, (Shape.Create(CX.Shape.AsArray)));
            executor.GetTensor(HY, (Shape.Create(HY.Shape.AsArray)));
            executor.GetTensor(CY, (Shape.Create(CY.Shape.AsArray)));

            if (IsTraining)
            {
                executor.GetGradient(X, (Shape.Create(X.Shape.AsArray)));
                executor.GetGradient(Y, (Shape.Create(Y.Shape.AsArray)));
                executor.GetGradient(HX, (Shape.Create(HX.Shape.AsArray)));
                executor.GetGradient(CX, (Shape.Create(CX.Shape.AsArray)));
                executor.GetGradient(HY, (Shape.Create(HY.Shape.AsArray)));
                executor.GetGradient(CY, (Shape.Create(CY.Shape.AsArray)));
            }

            // set LSTM weights
            var numLinearLayers = 8; // now we fixed it, hard code LSTM

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
                        int length;

                        deviceptr<T> linLayerMat;
                        dnn.GetRNNLinLayerMatrixParams(rnnDesc, layer, XDesc[0], wDesc, w.Buffer.Ptr, linLayerId,
                            filterDesc, out linLayerMat);

                        filterDesc.GetND(out dataType, out format, out nbDims, filterDimA);
                        length = filterDimA.Aggregate(ScalarOps.Mul);
                        //var value = 1.0/length;

                        var linLayerMatBuffer = new Buffer<T>(context.Device, w.Memory, new Layout(Shape.Create(length)),
                            linLayerMat);
                        var linLayerMatTensor = new Tensor<T>(linLayerMatBuffer);
                        //context.Assign(linLayerMatTensor, ScalarOps.Conv<T>(value));
                        context.Assign(linLayerMatTensor, RandomNormal<T>(Shape.Create(length))/(Math.Sqrt(HiddenSize+InputSize).AsScalar<T>()));

                        deviceptr<T> linLayerBias;
                        dnn.GetRNNLinLayerBiasParams(rnnDesc, layer, XDesc[0], wDesc, w.Buffer.Ptr, linLayerId,
                            filterDesc, out linLayerBias);

                        filterDesc.GetND(out dataType, out format, out nbDims, filterDimA);
                        length = filterDimA.Aggregate(ScalarOps.Mul);

                        var linLayerBiasBuffer = new Buffer<T>(context.Device, w.Memory, new Layout(Shape.Create(length)),
                            linLayerBias);
                        var linLayerBiasTensor = new Tensor<T>(linLayerBiasBuffer);
                        context.Assign(linLayerBiasTensor, ScalarOps.Conv<T>(Bias));
                    }
                }
            }

            base.Initialize(executor);

            const double value = 0.0;

            executor.AssignTensor(HX, AleaTK.Library.Fill(Shape.Create(HX.Shape.AsArray), ScalarOps.Conv<T>(value)));
            executor.AssignTensor(CX, AleaTK.Library.Fill(Shape.Create(CX.Shape.AsArray), ScalarOps.Conv<T>(value)));
            executor.AssignTensor(HY, AleaTK.Library.Fill(Shape.Create(HY.Shape.AsArray), ScalarOps.Conv<T>(value)));
            executor.AssignTensor(CY, AleaTK.Library.Fill(Shape.Create(HY.Shape.AsArray), ScalarOps.Conv<T>(value)));

            // we assign them directly (no gradient counter increasing)
            if (IsTraining)
            {
                executor.AssignGradientDirectly(HY, AleaTK.Library.Fill(Shape.Create(HY.Shape.AsArray), ScalarOps.Conv<T>(value)));
                executor.AssignGradientDirectly(CY, AleaTK.Library.Fill(Shape.Create(CY.Shape.AsArray), ScalarOps.Conv<T>(value)));
                executor.AssignGradientDirectly(W, ScalarOps.Conv<T>(0.0).AsScalar());
            }
        }

        public override void Forward(Executor executor)
        {
            var context = executor.Context.ToGpuContext();
            var dnn = context.Dnn;
            var rnnDesc = executor.RnnDescDict[RnnDesc];
            var seqLength = SeqLength;
            var xDesc = XDesc;
            var x = executor.GetTensor(X);
            var hxDesc = StateDesc;
            var hx = executor.GetTensor(HX);
            var cxDesc = StateDesc;
            var cx = executor.GetTensor(CX);
            var wDesc = executor.FilterDescDict[WDesc];
            var w = executor.GetTensor(W);
            var yDesc = YDesc;
            var y = executor.GetTensor(Y);
            var hyDesc = StateDesc;
            var hy = executor.GetTensor(HY);
            var cyDesc = StateDesc;
            var cy = executor.GetTensor(CY);
            var workspace = executor.GetTensor(Workspace);

            if (IsTraining)
            {
                //executor.AssignTensor(HX, hy);
                //executor.AssignTensor(CX, cy);
                executor.AssignTensor(HX, Fill(hx.Shape, ScalarOps.Conv<T>(0.0)));
                executor.AssignTensor(CX, Fill(cx.Shape, ScalarOps.Conv<T>(0.0)));

                //executor.Context.Eval(cx.Reshape(-1)).Print();

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
                    workspace.Buffer.Ptr, (IntPtr) workspace.Shape.Length);
            }
        }

        public override void Backward(Executor executor)
        {
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

            //executor.AssignGradientDirectly(HY, executor.GetGradient(HX));
            //executor.AssignGradientDirectly(CY, executor.GetGradient(CX));

            //executor.Context.Eval(executor.GetTensor(HY).Reshape(-1)).Print();

            dnn.RNNBackwardData(
                executor.RnnDescDict[RnnDesc],
                SeqLength,
                YDesc,
                executor.GetTensor(Y).Buffer.Ptr,
                YDesc,
                executor.GetGradient(Y).Buffer.Ptr,
                StateDesc,
                //executor.GetGradient(HY).Buffer.Ptr,
                new deviceptr<T>(), 
                StateDesc,
                //executor.GetGradient(CY).Buffer.Ptr,
                new deviceptr<T>(), 
                executor.FilterDescDict[WDesc],
                executor.GetTensor(W).Buffer.Ptr,
                StateDesc,
                executor.GetTensor(HX).Buffer.Ptr,
                StateDesc,
                executor.GetTensor(CX).Buffer.Ptr,
                XDesc,
                executor.GetGradient(X).Buffer.Ptr,
                StateDesc,
                executor.GetGradient(HX).Buffer.Ptr,
                StateDesc,
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
                SeqLength,
                XDesc,
                executor.GetTensor(X).Buffer.Ptr,
                StateDesc,
                executor.GetTensor(HX).Buffer.Ptr,
                YDesc,
                executor.GetTensor(Y).Buffer.Ptr,
                executor.GetTensor(Workspace).Buffer.Ptr,
                (IntPtr) executor.GetTensor(Workspace).Shape.Length,
                executor.FilterDescDict[WDesc],
                executor.GetGradient(W).Buffer.Ptr,
                executor.GetTensor(ReserveSpace).Buffer.Ptr,
                (IntPtr)executor.GetTensor(ReserveSpace).Shape.Length);
        }
    }
}
