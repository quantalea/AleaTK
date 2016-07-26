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
        public LSTM(Variable<T> x, int hiddenSize, Variable<T> cx = null, Variable<T> hx = null, double forgetBiasInit = 0.0)
        {
            // X shape (seqLength, batch, inputSize)
            Util.EnsureEqual(3, x.Shape.Rank, "Input layout: (seqLength, batch, inputSize)");
            Util.EnsureTrue(x.Shape[0] > 0, "SeqLength must be determined.");
            Util.EnsureTrue(x.Shape[2] > 0, "InputSize must be determined.");
            X = x;
            SeqLength = (int)X.Shape[0];
            InputSize = (int)X.Shape[2];
            HiddenSize = hiddenSize;
            ForgetBiasInit = forgetBiasInit;

            // Y Shape (seqLength, batch, hiddenSize)
            Y = Variable<T>(PartialShape.Create(SeqLength, -1, HiddenSize));

            // W (1 + inputSize + hiddenSize, 4 * hiddenSize) : B -> W -> U
            // layout: IFOA
            W =
                Parameter(RandomNormal<T>(Shape.Create(InputSize + HiddenSize + 1, 4*HiddenSize))/
                          (Math.Sqrt(InputSize + hiddenSize)).AsScalar<T>());
            // the following W initialization happens in Initialize();

            // input and output states
            CX = cx ?? Variable<T>(PartialShape.Create(-1, HiddenSize));
            HX = hx ?? Variable<T>(PartialShape.Create(-1, HiddenSize));
            CY = Variable<T>(PartialShape.Create(-1, HiddenSize));
            HY = Variable<T>(PartialShape.Create(-1, HiddenSize));

            // build the graph
            AddInput(X);
            AddOutput(Y);
            AddInput(W);
            AddInput(CX);
            AddInput(HX);
            AddOutput(CY);
            AddOutput(HY);

            // Aux variables
            Hin = AuxVariable<T>();
            Hout = AuxVariable<T>();
            IFOA1 = AuxVariable<T>();
            IFOA2 = AuxVariable<T>();
            C = AuxVariable<T>();
            Temp1 = AuxVariable<T>();

            AddAuxVar(Hin);
            AddAuxVar(Hout);
            AddAuxVar(IFOA1);
            AddAuxVar(IFOA2);
            AddAuxVar(C);
            AddAuxVar(Temp1);
        }

        public override void Initialize(Executor executor)
        {
            base.Initialize(executor);

            // set bias to zero
            var ctx = executor.Context;
            var w = executor.GetTensor(W);

            // first set 4 bias to 0.0
            ctx.Assign(w.Slice(0), 0.0.AsScalar<T>());

            // set forget bias is needed, layout: IFOA, so forget index is 1
            if (ForgetBiasInit != 0.0)
            {
                ctx.Assign(w.Slice(0, Range(HiddenSize, 2*HiddenSize)),
                    Fill(Shape.Create(1, HiddenSize), ScalarOps.Conv<T>(ForgetBiasInit)));
            }
        }

        public override void Forward(Executor executor)
        {
            var ctx = executor.Context;
            var w = executor.GetTensor(W);
            var xphpb = w.Shape[0];
            var x = executor.GetTensor(X);
            var b = x.Shape[1];
            var n = x.Shape[0];
            var d = HiddenSize;
            var y = executor.GetTensor(Y, Shape.Create(n, b, d));
            var inputSize = InputSize;
            var one = 1.0.AsScalar<T>();

            // inital states
            var cx = executor.GetTensor(CX);
            var hx = executor.GetTensor(HX);
            Util.EnsureTrue(cx.Shape.SequenceEqual(Shape.Create(b, d)));
            Util.EnsureTrue(hx.Shape.SequenceEqual(Shape.Create(b, d)));

            // we assign output states to inital states, and later we update it
            var cy = executor.GetTensor(CY, Shape.Create(b, d));
            var hy = executor.GetTensor(HY, Shape.Create(b, d));
            ctx.Assign(cy, cx);
            ctx.Assign(hy, hx);
            var prevc = cy.Reshape(1, b, d);
            var prevh = hy.Reshape(1, b, d);

            var hin = executor.GetTensor(Hin, Shape.Create(n, b, xphpb));
            var ifoa1 = executor.GetTensor(IFOA1, Shape.Create(n, b, d * 4));
            var ifoa2 = executor.GetTensor(IFOA2, Shape.Create(n, b, d * 4));
            var c = executor.GetTensor(C, Shape.Create(n, b, d));

            for (var t = 0; t < n; ++t)
            {
                // stack input
                ctx.Assign(hin.Slice(t, -1, 0), Fill(Shape.Create(1, b, 1), ScalarOps.Conv<T>(1.0))); // bias
                ctx.Assign(hin.Slice(t, -1, Range(1, inputSize + 1)), x.Slice(t));
                ctx.Assign(hin.Slice(t, -1, Range(inputSize + 1, -1)), prevh);

                // dot
                ctx.Assign(ifoa1.Slice(t), Dot(hin.Slice(t).Reshape(b, xphpb), w));

                // values for applying element-wise transformation
                // they are of shape (1, b, d)
                var ct = c.Slice(t);
                var ht = y.Slice(t);
                var it = ifoa2.Slice(t, -1, Range(0, d));
                var ft = ifoa2.Slice(t, -1, Range(d, 2 * d));
                var ot = ifoa2.Slice(t, -1, Range(2 * d, 3 * d));
                var at = ifoa2.Slice(t, -1, Range(3 * d, 4 * d));

                // non-linearities
                // first 3 matrices are IFO, we apply sigmoid
                var ifot = ifoa2.Slice(t, -1, Range(0, 3*d));
                var _ifot = ifoa1.Slice(t, -1, Range(0, 3*d));
                ctx.Assign(ifot, one/(one + Exp(-_ifot)));

                // last one is for activation gate, we apply tanh
                var _at = ifoa1.Slice(t, -1, Range(3*d, 4*d));
                ctx.Assign(at, Tanh(_at));

                // c_t = i_t * a_t + f_t * c_t-1
                ctx.Assign(ct, it*at + ft*prevc);

                // h_t = o_t * tanh(c_t)
                ctx.Assign(ht, ot* Tanh(ct));

                // update states
                ctx.Assign(prevh, y.Slice(t));
                ctx.Assign(prevc, c.Slice(t));
            }
        }

        public static Tensor<T> GetGradient(Executor executor, Variable<T> var, bool zerolize = false)
        {
            var ctx = executor.Context;
            var data = executor.GetData(var);
            Util.EnsureTrue(data.GradientAggregationCounter == 0);
            var shape = executor.GetTensor(var).Shape;
            var gradient = executor.GetGradient(var, shape);
            if (zerolize) ctx.Assign(gradient, Fill(shape, ScalarOps.Conv<T>(0.0)));
            return gradient;
        }

        public static Tensor<T> GetGradient(Executor executor, Variable<T> var, Shape shape, bool zerolize = false)
        {
            var ctx = executor.Context;
            var data = executor.GetData(var);
            Util.EnsureTrue(data.GradientAggregationCounter == 0);
            var gradient = executor.GetGradient(var, shape);
            if (zerolize) ctx.Assign(gradient, Fill(shape, ScalarOps.Conv<T>(0.0)));
            return gradient;
        }

        public override void Backward(Executor executor)
        {
            var ctx = executor.Context;
            var one = 1.0.AsScalar<T>();

            var dy = executor.GetGradient(Y); // input
            var w = executor.GetTensor(W);
            var c = executor.GetTensor(C);
            var hin = executor.GetTensor(Hin);
            var ifoa2 = executor.GetTensor(IFOA2);
            var n = dy.Shape[0];
            var b = dy.Shape[1];
            var d = (int)dy.Shape[2];
            var xphpb = w.Shape[0];
            var inputSize = InputSize;

            var cx = executor.GetTensor(CX);
            var hx = executor.GetTensor(HX);
            Util.EnsureTrue(cx.Shape.SequenceEqual(Shape.Create(b, d)));
            Util.EnsureTrue(hx.Shape.SequenceEqual(Shape.Create(b, d)));

            var dc = GetGradient(executor, C, zerolize: true);
            var dx = executor.GetGradient(X, Shape.Create(n, b, inputSize));
            var dw = GetGradient(executor, W, zerolize: true);
            var difoa1 = GetGradient(executor, IFOA1, ifoa2.Shape);
            var difoa2 = GetGradient(executor, IFOA2, ifoa2.Shape);
            var dhin = GetGradient(executor, Hin, zerolize: true);
            var dhout = GetGradient(executor, Hout, dy.Shape, zerolize: true);
            var dhx = GetGradient(executor, HX, zerolize: true);
            var dcx = GetGradient(executor, CX, zerolize: true);

            ctx.Assign(dhout, dy);

            // TODO: dcn and dhn
            // now all are 0!

            for (var t = n - 1; t >= 0; --t)
            {
                var dct = dc.Slice(t);
                var dht = dhout.Slice(t);
                var dit = difoa2.Slice(t, -1, Range(0, d));
                var dft = difoa2.Slice(t, -1, Range(d, 2*d));
                var dot = difoa2.Slice(t, -1, Range(2*d, 3*d));
                var dat = difoa2.Slice(t, -1, Range(3*d, 4*d));

                var it = ifoa2.Slice(t, -1, Range(0, d));
                var ft = ifoa2.Slice(t, -1, Range(d, 2*d));
                var ot = ifoa2.Slice(t, -1, Range(2*d, 3*d));
                var at = ifoa2.Slice(t, -1, Range(3*d, 4*d));
                var ct = c.Slice(t);

                // do_t = dh_t * tanh(c_t)
                ctx.Assign(dot, dht*Tanh(ct));

                // dc_t += dh_t * o_t * (1 - tanh**2(c_t))
                ctx.Assign(dct, dct + dht*ot*(one - Tanh(ct)*Tanh(ct)));

                // df_t = dc_t * c_t-1
                // dc_t-1 = dc_t * f_t
                if (t > 0)
                {
                    var ctPrev = c.Slice(t - 1);
                    var dctPrev = dc.Slice(t - 1);
                    ctx.Assign(dft, dct*ctPrev);
                    // in-place add, because dcy might not be 0
                    ctx.Assign(dctPrev, dctPrev + ft * dct);
                }
                else
                {
                    var ctPrev = cx;
                    var dctPrev = dcx;
                    ctx.Assign(dft, dct*ctPrev);
                    ctx.Assign(dctPrev, (ft*dct).Reshape(b, d));
                }
                // di_t = dc_t * a_t
                ctx.Assign(dit, dct*at);
                // da_t = dc_t * i_t
                ctx.Assign(dat, dct*it);

                // backprop activation functions
                // d^a_t = (1 - a_t * a_t) * da_t
                var _dat = difoa1.Slice(t, -1, Range(3*d, 4*d));
                ctx.Assign(_dat, (one - at*at)*dat);

                // d_sigmoid for other 3 matrices : d^ifo_t = ifo_t * (1 - ifo_t) * difo_t
                var ifot = ifoa2.Slice(t, -1, Range(0, 3*d));
                var difot = difoa2.Slice(t, -1, Range(0, 3*d));
                var _difot = difoa1.Slice(t, -1, Range(0, 3*d));
                ctx.Assign(_difot, ifot*(one - ifot)*difot);

                // backprop matrix multiply
                var _difoat = difoa1.Slice(t).Reshape(b, 4*d);
                var tmp = executor.GetTensor(Temp1, Shape.Create(b, xphpb));
                ctx.Assign(tmp, hin.Slice(t).Reshape(b, xphpb));
                ctx.Assign(dw, dw + Dot(tmp.T, _difoat));
                ctx.Assign(dhin.Slice(t), Dot(_difoat, w.T));

                // backprop the identity transforms into hin
                var dxt = dx.Slice(t);
                ctx.Assign(dxt, dhin.Slice(t, -1, Range(1, InputSize + 1)));
                if (t > 0)
                {
                    ctx.Assign(dhout.Slice(t - 1), dhout.Slice(t - 1) + dhin.Slice(t, -1, Range(InputSize + 1, -1)));
                }
                else
                {
                    ctx.Assign(dhx, (dhx.Reshape(1, b, d) + dhin.Slice(t, -1, Range(InputSize + 1, -1))).Reshape(b, d));
                }
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

        public Variable<T> CY { get; }

        public Variable<T> HY { get; }

        public Variable<T> Hin { get; }

        public Variable<T> Hout { get; }

        public Variable<T> IFOA1 { get; }

        public Variable<T> IFOA2 { get; }

        public Variable<T> C { get; }

        public Variable<T> Temp1 { get; }

    }

    public class LSTM1b<T> : Differentiable
    {
        public LSTM1b(Variable<T> x, int hiddenSize, Variable<T> cx = null, Variable<T> hx = null, double forgetBiasInit = 3.0)
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
                Parameter(RandomNormal<T>(Shape.Create(InputSize + HiddenSize + 1, 4 * HiddenSize)) /
                          (Math.Sqrt(InputSize + hiddenSize)).AsScalar<T>());
            // the following W initialization happens in Initialize();

            Hin = AuxVariable<T>();
            Hout = AuxVariable<T>();
            IFOG = AuxVariable<T>();
            IFOGf = AuxVariable<T>();
            C = AuxVariable<T>();
            Ct = AuxVariable<T>();
            Temp1 = AuxVariable<T>();
            CX = cx ?? Variable<T>(PartialShape.Create(-1, HiddenSize));
            HX = hx ?? Variable<T>(PartialShape.Create(-1, HiddenSize));
            CY = Variable<T>(PartialShape.Create(-1, HiddenSize));
            HY = Variable<T>(PartialShape.Create(-1, HiddenSize));

            AddInput(X);
            AddOutput(Y);
            AddInput(W);
            AddInput(CX);
            AddInput(HX);
            AddOutput(CY);
            AddOutput(HY);
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
                ctx.Assign(w.Slice(0, Range(HiddenSize, HiddenSize * 2)),
                    Fill(Shape.Create(1, HiddenSize), ScalarOps.Conv<T>(ForgetBiasInit)));
            }
        }

        public override void Forward(Executor executor)
        {
            var ctx = executor.Context;
            Util.EnsureTrue(ctx.Type == ContextType.Gpu);
            var stream = ctx.ToGpuContext().Stream;

            var w = executor.GetTensor(W);
            var xphpb = w.Shape[0];
            var x = executor.GetTensor(X);
            var b = x.Shape[1];
            var n = x.Shape[0];
            var d = HiddenSize;

            var c0 = executor.GetTensor(CX);
            var h0 = executor.GetTensor(HX);
            var cn = executor.GetTensor(CY, Shape.Create(b, d));
            var hn = executor.GetTensor(HY, Shape.Create(b, d));
            Util.EnsureTrue(c0.Shape.SequenceEqual(Shape.Create(b, d)));
            Util.EnsureTrue(h0.Shape.SequenceEqual(Shape.Create(b, d)));

            var hin = executor.GetTensor(Hin, Shape.Create(n, b, xphpb));
            var hout = executor.GetTensor(Hout, Shape.Create(n, b, d));
            var ifog = executor.GetTensor(IFOG, Shape.Create(n, b, d * 4));
            var ifogf = executor.GetTensor(IFOGf, Shape.Create(n, b, d * 4));
            var c = executor.GetTensor(C, Shape.Create(n, b, d));
            var ct = executor.GetTensor(Ct, Shape.Create(n, b, d));

            var prevh = hn.Reshape(1, b, d);
            var prevc = cn.Reshape(1, b, d);

            for (var t = 0; t < n; ++t)
            {
                // stack input
                ctx.Assign(prevh, t > 0 ? hout.Slice(t - 1) : h0);
                ctx.Assign(hin.Slice(t, -1, 0), Fill(Shape.Create(1, b, 1), ScalarOps.Conv<T>(1.0))); // bias
                ctx.Assign(hin.Slice(t, -1, Range(1, InputSize + 1)), x.Slice(t));
                ctx.Assign(hin.Slice(t, -1, Range(InputSize + 1, -1)), prevh);

                // dot
                ctx.Assign(ifog.Slice(t), Dot(hin.Slice(t).Reshape(b, xphpb), w));

                // non-linearities
                // first 3 matrices are ifo
                ctx.Assign(ifogf.Slice(t, -1, Range(0, 3 * d)),
                    1.0.AsScalar<T>() / (1.0.AsScalar<T>() + Exp(-ifog.Slice(t, -1, Range(0, 3 * d)))));

                // last one is for g(a)
                ctx.Assign(ifogf.Slice(t, -1, Range(3 * d, -1)), Tanh(ifog.Slice(t, -1, Range(3 * d, -1))));

                // update c
                ctx.Assign(prevc, t > 0 ? c.Slice(t - 1) : c0);
                // c_t = i_t * a_t + f_t * c_t-1
                ctx.Assign(c.Slice(t),
                    ifogf.Slice(t, -1, Range(0, d)) * ifogf.Slice(t, -1, Range(3 * d, -1)) +
                    ifogf.Slice(t, -1, Range(d, 2 * d)) * prevc);
                // h_t = o_t * tanh(c_t)
                ctx.Assign(ct.Slice(t), Tanh(c.Slice(t)));
                ctx.Assign(hout.Slice(t), ifogf.Slice(t, -1, Range(2 * d, 3 * d)) * ct.Slice(t));
            }

            ctx.Assign(prevc, c.Slice(n - 1));
            ctx.Assign(prevh, hout.Slice(n - 1));
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
            var xphpb = w.Shape[0];

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
                ctx.Assign(dIFOGf.Slice(t, -1, Range(2 * d, 3 * d)), tanhCt * dhout.Slice(t));

                // dc_t += dh_t * o_t * (1 - tanh**2(c_t))
                ctx.Assign(dc.Slice(t),
                    dc.Slice(t) +
                    (1.0.AsScalar<T>() - tanhCt * tanhCt) * (ifogf.Slice(t, -1, Range(2 * d, 3 * d)) * dhout.Slice(t)));

                // df_t = dc_t * c_t-1
                if (t > 0)
                {
                    ctx.Assign(dIFOGf.Slice(t, -1, Range(d, 2 * d)), c.Slice(t - 1) * dc.Slice(t));
                    ctx.Assign(dc.Slice(t - 1), dc.Slice(t - 1) + ifogf.Slice(t, -1, Range(d, 2 * d)) * dc.Slice(t));
                }
                else
                {
                    ctx.Assign(dIFOGf.Slice(t, -1, Range(d, 2 * d)), c0 * dc.Slice(t));
                    ctx.Assign(dc0, (ifogf.Slice(t, -1, Range(d, 2 * d)) * dc.Slice(t)).Reshape(b, d));
                }
                // di_t = dc_t * a_t
                ctx.Assign(dIFOGf.Slice(t, -1, Range(0, d)), ifogf.Slice(t, -1, Range(3 * d, -1)) * dc.Slice(t));
                // da_t = dc_t * i_t
                ctx.Assign(dIFOGf.Slice(t, -1, Range(3 * d, -1)), ifogf.Slice(t, -1, Range(0, d)) * dc.Slice(t));

                // backprop activation functions
                var tmp1 = ifogf.Slice(t, -1, Range(3 * d, -1));
                ctx.Assign(dIFOG.Slice(t, -1, Range(3 * d, -1)), (1.0.AsScalar<T>() - tmp1 * tmp1) * dIFOGf.Slice(t, -1, Range(3 * d, -1)));
                var tmp2 = ifogf.Slice(t, -1, Range(0, 3 * d));
                ctx.Assign(dIFOG.Slice(t, -1, Range(0, 3 * d)),
                    (tmp2 * (1.0.AsScalar<T>() - tmp2)) * dIFOGf.Slice(t, -1, Range(0, 3 * d)));

                // backprop matrix multiply
                var tmp3 = executor.GetTensor(Temp1, Shape.Create(b, xphpb));
                ctx.Assign(tmp3, hin.Slice(t).Reshape(b, xphpb));
                ctx.Assign(dw, dw + Dot(tmp3.T, dIFOG.Slice(t).Reshape(b, 4 * d)));
                ctx.Assign(dhin.Slice(t), Dot(dIFOG.Slice(t).Reshape(b, 4 * d), w.T));

                // backprop the identity transforms into hin
                ctx.Assign(dx.Slice(t), dhin.Slice(t, -1, Range(1, InputSize + 1)));
                if (t > 0)
                {
                    ctx.Assign(dhout.Slice(t - 1), dhout.Slice(t - 1) + dhin.Slice(t, -1, Range(InputSize + 1, -1)));
                }
                else
                {
                    ctx.Assign(dh0, (dh0.Reshape(1, b, d) + dhin.Slice(t, -1, Range(InputSize + 1, -1))).Reshape(b, d));
                }
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

        public Variable<T> CY { get; }

        public Variable<T> HY { get; }

        public Variable<T> Hin { get; }

        public Variable<T> Hout { get; }

        public Variable<T> IFOG { get; }

        public Variable<T> IFOGf { get; }

        public Variable<T> C { get; }

        public Variable<T> Ct { get; }

        public Variable<T> Temp1 { get; }
    }

    public class LSTM2<T> : Differentiable
    {
        public LSTM2(Variable<T> x, int hiddenSize, double forgetBiasInit = 3.0, Variable<T> cx = null, Variable<T> hx = null)
        {
            // X shape (seqLength, batch, inputSize)
            Util.EnsureEqual(3, x.Shape.Rank, "Input layout: (seqLength, batch, inputSize)");
            Util.EnsureTrue(X.Shape[0] > 0, "SeqLength must be determined.");
            Util.EnsureTrue(X.Shape[2] > 0, "InputSize must be determined.");
            X = x;
            SeqLength = (int)X.Shape[0];
            InputSize = (int)X.Shape[2];
            HiddenSize = hiddenSize;
            ForgetBiasInit = forgetBiasInit;

            // Y Shape (seqLength, batch, hiddenSize)
            Y = Variable<T>(PartialShape.Create(SeqLength, -1, HiddenSize));

            // W (inputSize + hiddenSize + 1, 4 * hiddenSize)
            // W (4 * inputSize + 4 * hiddenSize + 4, hiddenSize)
            // our W layout is AIFO
            W =
                Parameter(RandomNormal<T>(Shape.Create(4 * InputSize + 4 * HiddenSize + 4, HiddenSize)) /
                          (Math.Sqrt(InputSize + hiddenSize)).AsScalar<T>());
            // the following W initialization happens in Initialize();

            CX = cx ?? Variable<T>(PartialShape.Create(-1, HiddenSize));
            HX = hx ?? Variable<T>(PartialShape.Create(-1, HiddenSize));
            CY = Variable<T>(PartialShape.Create(-1, HiddenSize));
            HY = Variable<T>(PartialShape.Create(-1, HiddenSize));

            AIFO1 = AuxVariable<T>();
            AIFO2 = AuxVariable<T>();
            C1 = AuxVariable<T>();
            C2 = AuxVariable<T>();

            AddInput(X);
            AddInput(W);
            AddInput(CX);
            AddInput(HX);
            AddOutput(Y);
            AddOutput(CY);
            AddOutput(HY);
            AddAuxVar(AIFO1);
            AddAuxVar(AIFO2);
            AddAuxVar(C1);
            AddAuxVar(C2);

            // --------------------------------

            Hin = AuxVariable<T>();
            Hout = AuxVariable<T>();
            IFOG = AuxVariable<T>();
            IFOGf = AuxVariable<T>();
            C = AuxVariable<T>();
            Ct = AuxVariable<T>();
            Temp1 = AuxVariable<T>();

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

            var ctx = executor.Context;
            var w = executor.GetTensor(W);

            // first set them all to 0
            // w layout: W -> U -> B, AIFO, so we locate the bias
            ctx.Assign(w.Slice(Range(4*InputSize + 4*HiddenSize, -1)), 0.0.AsScalar<T>());

            // set forget bias if possible, remember: AIFO
            if (ForgetBiasInit != 0.0)
            {
                ctx.Assign(w.Slice(4*InputSize + 4*HiddenSize + 2), ForgetBiasInit.AsScalar<T>());
            }
        }

        public override void Forward(Executor executor)
        {
            var ctx = executor.Context;
            var one = 1.0.AsScalar<T>();

            // W layout: AIFO
            var oftU = 4 * InputSize;
            var oftB = 4 * InputSize + 4 * HiddenSize;

            var w = executor.GetTensor(W);
            var x = executor.GetTensor(X);
            var inputSize = InputSize;
            var hiddenSize = HiddenSize;
            var n = x.Shape[0];
            var b = x.Shape[1];
            var d = x.Shape[2];
            var y = executor.GetTensor(Y, Shape.Create(n, b, d));

            var cx = executor.GetTensor(CX);
            var hx = executor.GetTensor(HX);
            Util.EnsureTrue(cx.Shape.SequenceEqual(Shape.Create(b, d)));
            Util.EnsureTrue(hx.Shape.SequenceEqual(Shape.Create(b, d)));

            var cy = executor.GetTensor(CY, Shape.Create(b, d));
            var hy = executor.GetTensor(HY, Shape.Create(b, d));
            ctx.Assign(cy, cx);
            ctx.Assign(hy, hx);

            // w matrices (AIFO)
            var oft = 0;
            var wa = w.Slice(Range(oft, oft + inputSize)); oft += inputSize;
            var wi = w.Slice(Range(oft, oft + inputSize)); oft += inputSize;
            var wf = w.Slice(Range(oft, oft + inputSize)); oft += inputSize;
            var wo = w.Slice(Range(oft, oft + inputSize));

            // u matrices (AIFO)
            oft = oftU;
            var ua = w.Slice(Range(oft, oft + hiddenSize)); oft += hiddenSize;
            var ui = w.Slice(Range(oft, oft + hiddenSize)); oft += hiddenSize;
            var uf = w.Slice(Range(oft, oft + hiddenSize)); oft += hiddenSize;
            var uo = w.Slice(Range(oft, oft + hiddenSize));

            // b vectors (AIFO)
            oft = oftB;
            var ba = w.Slice(oft); oft++;
            var bi = w.Slice(oft); oft++;
            var bf = w.Slice(oft); oft++;
            var bo = w.Slice(oft);

            // aifo 1 and 2, for storing intermediate value of aifo
            // layout is (4, n, b, d) for AIFO
            var aifo1 = executor.GetTensor(AIFO1, Shape.Create(4, n, b, d));
            var a1 = aifo1.Slice(0).Reshape(n, b, d);
            var i1 = aifo1.Slice(1).Reshape(n, b, d);
            var f1 = aifo1.Slice(2).Reshape(n, b, d);
            var o1 = aifo1.Slice(3).Reshape(n, b, d);

            var aifo2 = executor.GetTensor(AIFO2, Shape.Create(4, n, b, d));
            var a2 = aifo2.Slice(0).Reshape(n, b, d);
            var i2 = aifo2.Slice(1).Reshape(n, b, d);
            var f2 = aifo2.Slice(2).Reshape(n, b, d);
            var o2 = aifo2.Slice(3).Reshape(n, b, d);

            // c1 and c2 are aux var
            var c1 = executor.GetTensor(C1, Shape.Create(n, b, d));
            var c2 = executor.GetTensor(C2, Shape.Create(n, b, d));

            // now start iteration
            for (var t = 0; t < n; ++t)
            {
                // slice xt (1, b, inputSize)
                var xt = x.Slice(t);
                var yt = y.Slice(t);

                // slice aifo;
                var at1 = a1.Slice(t);
                var it1 = i1.Slice(t);
                var ft1 = f1.Slice(t);
                var ot1 = o1.Slice(t);

                var at2 = a2.Slice(t);
                var it2 = i2.Slice(t);
                var ft2 = f2.Slice(t);
                var ot2 = o2.Slice(t);

                var ct1 = c1.Slice(t);
                var ct2 = c2.Slice(t);

                // dot W
                ctx.Assign(at1, Dot(xt.Reshape(b, inputSize), wa));
                ctx.Assign(it1, Dot(xt.Reshape(b, inputSize), wi));
                ctx.Assign(ft1, Dot(xt.Reshape(b, inputSize), wf));
                ctx.Assign(ot1, Dot(xt.Reshape(b, inputSize), wo));

                // dot U
                ctx.Assign(at2, Dot(hy, ua));
                ctx.Assign(it2, Dot(hy, ui));
                ctx.Assign(ft2, Dot(hy, uf));
                ctx.Assign(ot2, Dot(hy, uo));

                // add together, store in aifo1
                ctx.Assign(at1, at1 + at2 + ba);
                ctx.Assign(it1, it1 + it2 + bi);
                ctx.Assign(ft1, ft1 + ft2 + bf);
                ctx.Assign(ot1, ot1 + ot2 + bo);

                // apply non-linear
                ctx.Assign(at2, Tanh(at1));
                ctx.Assign(it2, one / (one + Exp(-it1)));
                ctx.Assign(ft2, one / (one + Exp(-ft1)));
                ctx.Assign(ot2, one / (one + Exp(-ot1)));

                // update c
                ctx.Assign(ct1, it2*at2 + ft2*cy);
                ctx.Assign(ct2, Tanh(ct1));
                ctx.Assign(yt, ot2*ct2);

                // update hy and cy
                ctx.Assign(cy, ct1);
                ctx.Assign(hy, yt);
            }
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
            var xphpb = w.Shape[0];

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
                ctx.Assign(dIFOGf.Slice(t, -1, Range(2 * d, 3 * d)), tanhCt * dhout.Slice(t));

                // dc_t += dh_t * o_t * (1 - tanh**2(c_t))
                ctx.Assign(dc.Slice(t),
                    dc.Slice(t) +
                    (1.0.AsScalar<T>() - tanhCt * tanhCt) * (ifogf.Slice(t, -1, Range(2 * d, 3 * d)) * dhout.Slice(t)));

                // df_t = dc_t * c_t-1
                if (t > 0)
                {
                    ctx.Assign(dIFOGf.Slice(t, -1, Range(d, 2 * d)), c.Slice(t - 1) * dc.Slice(t));
                    ctx.Assign(dc.Slice(t - 1), dc.Slice(t - 1) + ifogf.Slice(t, -1, Range(d, 2 * d)) * dc.Slice(t));
                }
                else
                {
                    ctx.Assign(dIFOGf.Slice(t, -1, Range(d, 2 * d)), c0 * dc.Slice(t));
                    ctx.Assign(dc0, (ifogf.Slice(t, -1, Range(d, 2 * d)) * dc.Slice(t)).Reshape(b, d));
                }
                // di_t = dc_t * a_t
                ctx.Assign(dIFOGf.Slice(t, -1, Range(0, d)), ifogf.Slice(t, -1, Range(3 * d, -1)) * dc.Slice(t));
                // da_t = dc_t * i_t
                ctx.Assign(dIFOGf.Slice(t, -1, Range(3 * d, -1)), ifogf.Slice(t, -1, Range(0, d)) * dc.Slice(t));

                // backprop activation functions
                var tmp1 = ifogf.Slice(t, -1, Range(3 * d, -1));
                ctx.Assign(dIFOG.Slice(t, -1, Range(3 * d, -1)), (1.0.AsScalar<T>() - tmp1 * tmp1) * dIFOGf.Slice(t, -1, Range(3 * d, -1)));
                var tmp2 = ifogf.Slice(t, -1, Range(0, 3 * d));
                ctx.Assign(dIFOG.Slice(t, -1, Range(0, 3 * d)),
                    (tmp2 * (1.0.AsScalar<T>() - tmp2)) * dIFOGf.Slice(t, -1, Range(0, 3 * d)));

                // backprop matrix multiply
                var tmp3 = executor.GetTensor(Temp1, Shape.Create(b, xphpb));
                ctx.Assign(tmp3, hin.Slice(t).Reshape(b, xphpb));
                ctx.Assign(dw, dw + Dot(tmp3.T, dIFOG.Slice(t).Reshape(b, 4 * d)));
                ctx.Assign(dhin.Slice(t), Dot(dIFOG.Slice(t).Reshape(b, 4 * d), w.T));

                // backprop the identity transforms into hin
                ctx.Assign(dx.Slice(t), dhin.Slice(t, -1, Range(1, InputSize + 1)));
                if (t > 0)
                {
                    ctx.Assign(dhout.Slice(t - 1), dhout.Slice(t - 1) + dhin.Slice(t, -1, Range(InputSize + 1, -1)));
                }
                else
                {
                    ctx.Assign(dh0, (dh0.Reshape(1, b, d) + dhin.Slice(t, -1, Range(InputSize + 1, -1))).Reshape(b, d));
                }
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

        public Variable<T> CY { get; }

        public Variable<T> HY { get; }

        public Variable<T> AIFO1 { get; }

        public Variable<T> AIFO2 { get; }

        public Variable<T> C1 { get; }

        public Variable<T> C2 { get; }

        // -------------------------

        public Variable<T> Hin { get; }

        public Variable<T> Hout { get; }

        public Variable<T> IFOG { get; }

        public Variable<T> IFOGf { get; }

        public Variable<T> C { get; }

        public Variable<T> Ct { get; }

        public Variable<T> Temp1 { get; }
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
            Dropout = isTraining ? dropout : 0.0;
            DropoutSeed = dropoutSeed;
            Util.EnsureTrue(bias == 0.0, "bias need TODO");

            // X shape (seqLength, batch, inputSize)
            Util.EnsureEqual(3, X.Shape.Rank, "Input layout: (seqLength, batch, inputSize)");
            Util.EnsureTrue(X.Shape[0] >= 0, "Input layout: (seqLength, batch, inputSize)");
            Util.EnsureTrue(X.Shape[1] >= 0, "Input layout: (seqLength, batch, inputSize)");
            Util.EnsureTrue(X.Shape[2] >= 0, "Input layout: (seqLength, batch, inputSize)");
            SeqLength = (int) X.Shape[0];
            MiniBatch = (int) X.Shape[1];
            InputSize = (int) X.Shape[2];

            // Y Shape (seqLength, batch, hiddenSize)
            Y = Variable<T>(PartialShape.Create(SeqLength, MiniBatch, HiddenSize));

            // W shape will be determined during initialization
            W = Parameter<T>();

            // state variables
            var shape = PartialShape.Create(NumLayers, MiniBatch, HiddenSize);
            var strides = Strides.Create(shape[1]*shape[2], shape[2], 1); // inner change most
            HX = Variable<T>(shape);
            CX = Variable<T>(shape);
            HY = Variable<T>(shape);
            CY = Variable<T>(shape);
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
            //Console.WriteLine($"DROPOUT: {Dropout}");

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
            }

            // set LSTM weights
            var numLinearLayers = 8; // now we fixed it, hard code LSTM

            using (var filterDesc = new FilterDescriptor())
            {
                var w = executor.GetTensor(W);
                //Console.WriteLine($"w: {w.Buffer.Ptr.Handle}");
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
                        //var offset = (linLayerMatTensor.Buffer.Ptr.Handle.ToInt64() - w.Buffer.Ptr.Handle.ToInt64())/Gpu.SizeOf<T>();
                        //Console.WriteLine($"w.{layer}.{linLayerId}: {offset} {nbDims} {Shape.Create(filterDimA.Select(ll => (long)ll).ToArray())} {dataType} {format}");
                        //context.Assign(linLayerMatTensor, ScalarOps.Conv<T>(value));
                        context.Assign(linLayerMatTensor, RandomNormal<T>(Shape.Create(length))/(Math.Sqrt(HiddenSize+InputSize).AsScalar<T>()));
                        //context.Assign(linLayerMatTensor, RandomUniform<T>(Shape.Create(length)) *0.1.AsScalar<T>() - 0.05.AsScalar<T>());
                        //context.Assign(linLayerMatTensor, 0.1.AsScalar<T>());

                        deviceptr<T> linLayerBias;
                        dnn.GetRNNLinLayerBiasParams(rnnDesc, layer, XDesc[0], wDesc, w.Buffer.Ptr, linLayerId,
                            filterDesc, out linLayerBias);

                        filterDesc.GetND(out dataType, out format, out nbDims, filterDimA);
                        length = filterDimA.Aggregate(ScalarOps.Mul);

                        var linLayerBiasBuffer = new Buffer<T>(context.Device, w.Memory, new Layout(Shape.Create(length)),
                            linLayerBias);
                        var linLayerBiasTensor = new Tensor<T>(linLayerBiasBuffer);
                        //offset = (linLayerBiasTensor.Buffer.Ptr.Handle.ToInt64() - w.Buffer.Ptr.Handle.ToInt64())/Gpu.SizeOf<T>();
                        //Console.WriteLine($"b.{layer}.{linLayerId}: {offset} {nbDims} {Shape.Create(filterDimA.Select(ll => (long)ll).ToArray())} {dataType} {format}");
                        // TODO: need check, there are 8 matrices, but usually you only need set 4 of them
                        context.Assign(linLayerBiasTensor, ScalarOps.Conv<T>(Bias));
                        //context.Assign(linLayerBiasTensor, 0.0.AsScalar<T>());
                    }
                }
            }

            base.Initialize(executor);

            const double value = 0.0;
            executor.AssignTensor(HX, Fill(Shape.Create(HX.Shape.AsArray), ScalarOps.Conv<T>(value)));
            executor.AssignTensor(CX, Fill(Shape.Create(CX.Shape.AsArray), ScalarOps.Conv<T>(value)));
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
