using System;
using System.Linq;
using Alea;
using Alea.Parallel;
using AleaTK;
using AleaTK.ML;
using static AleaTK.Library;
using static AleaTK.ML.Library;

namespace Tutorial.Samples
{
    /// <summary>
    /// Direct implementation of a single Lstm layer following Karpathy's implementation
    /// https://gist.github.com/karpathy/587454dc0146a6ae21fc
    /// </summary>
    public class Lstm<T> : Differentiable
    {
        public Lstm(Variable<T> x, int hiddenSize, Variable<T> cx = null, Variable<T> hx = null, double forgetBiasInit = 0.0)
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
            W = Parameter(RandomNormal<T>(Shape.Create(InputSize + HiddenSize + 1, 4 * HiddenSize)) / Math.Sqrt(InputSize + hiddenSize).AsScalar<T>());

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
            Temp2 = AuxVariable<T>();

            AddAuxVar(Hin);
            AddAuxVar(Hout);
            AddAuxVar(IFOA1);
            AddAuxVar(IFOA2);
            AddAuxVar(C);
            AddAuxVar(Temp1);
            AddAuxVar(Temp2);
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
                ctx.Assign(w.Slice(0, Range(HiddenSize, 2 * HiddenSize)), Fill(Shape.Create(1, HiddenSize), ScalarOps.Conv<T>(ForgetBiasInit)));
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

        public const bool UseOptimizedImplementation = true;

        public override void Forward(Executor executor)
        {
            if (UseOptimizedImplementation)
            {
                ForwardOptimized(executor);
            }
            else
            {
                ForwardBasic(executor);
            }
        }

        public override void Backward(Executor executor)
        {
            if (UseOptimizedImplementation)
            {
                BackwardOptimized(executor);
            }
            else
            {
                BackwardBasic(executor);
            }
        }

        public void ForwardBasic(Executor executor)
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
                var ifot = ifoa2.Slice(t, -1, Range(0, 3 * d));
                var _ifot = ifoa1.Slice(t, -1, Range(0, 3 * d));
                ctx.Assign(ifot, one / (one + Exp(-_ifot)));

                // last one is for activation gate, we apply tanh
                var _at = ifoa1.Slice(t, -1, Range(3 * d, 4 * d));
                ctx.Assign(at, Tanh(_at));

                // c_t = i_t * a_t + f_t * c_t-1
                ctx.Assign(ct, it * at + ft * prevc);

                // h_t = o_t * tanh(c_t)
                ctx.Assign(ht, ot * Tanh(ct));

                // update states
                ctx.Assign(prevh, y.Slice(t));
                ctx.Assign(prevc, c.Slice(t));
            }
        }

        public void BackwardBasic(Executor executor)
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

            for (var t = n - 1; t >= 0; --t)
            {
                var dct = dc.Slice(t);
                var dht = dhout.Slice(t);
                var dit = difoa2.Slice(t, -1, Range(0, d));
                var dft = difoa2.Slice(t, -1, Range(d, 2 * d));
                var dot = difoa2.Slice(t, -1, Range(2 * d, 3 * d));
                var dat = difoa2.Slice(t, -1, Range(3 * d, 4 * d));

                var it = ifoa2.Slice(t, -1, Range(0, d));
                var ft = ifoa2.Slice(t, -1, Range(d, 2 * d));
                var ot = ifoa2.Slice(t, -1, Range(2 * d, 3 * d));
                var at = ifoa2.Slice(t, -1, Range(3 * d, 4 * d));
                var ct = c.Slice(t);

                // do_t = dh_t * tanh(c_t)
                ctx.Assign(dot, dht * Tanh(ct));

                // dc_t += dh_t * o_t * (1 - tanh**2(c_t))
                ctx.Assign(dct, dct + dht * ot * (one - Tanh(ct) * Tanh(ct)));

                // df_t = dc_t * c_t-1
                // dc_t-1 = dc_t * f_t
                if (t > 0)
                {
                    var ctPrev = c.Slice(t - 1);
                    var dctPrev = dc.Slice(t - 1);
                    ctx.Assign(dft, dct * ctPrev);
                    // in-place add, because dcy might not be 0
                    ctx.Assign(dctPrev, dctPrev + ft * dct);
                }
                else
                {
                    var ctPrev = cx;
                    var dctPrev = dcx;
                    ctx.Assign(dft, dct * ctPrev);
                    ctx.Assign(dctPrev, (ft * dct).Reshape(b, d));
                }
                // di_t = dc_t * a_t
                ctx.Assign(dit, dct * at);
                // da_t = dc_t * i_t
                ctx.Assign(dat, dct * it);

                // backprop activation functions
                // d^a_t = (1 - a_t * a_t) * da_t
                var _dat = difoa1.Slice(t, -1, Range(3 * d, 4 * d));
                ctx.Assign(_dat, (one - at * at) * dat);

                // d_sigmoid for other 3 matrices : d^ifo_t = ifo_t * (1 - ifo_t) * difo_t
                var ifot = ifoa2.Slice(t, -1, Range(0, 3 * d));
                var difot = difoa2.Slice(t, -1, Range(0, 3 * d));
                var _difot = difoa1.Slice(t, -1, Range(0, 3 * d));
                ctx.Assign(_difot, ifot * (one - ifot) * difot);

                // backprop matrix multiply
                var _difoat = difoa1.Slice(t).Reshape(b, 4 * d);
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

        public void ForwardOptimized(Executor executor)
        {
            var ctx = executor.Context;
            var w = executor.GetTensor(W);
            var xphpb = (int)w.Shape[0];
            var x = executor.GetTensor(X);
            var b = (int)x.Shape[1];
            var n = (int)x.Shape[0];
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

            Util.EnsureTrue(ctx.Type == ContextType.Gpu && typeof(T) == typeof(float), "Currently only support gpu and single precision.");

            if (ctx.Type == ContextType.Gpu && typeof(T) == typeof(float))
            {
                var stream = ctx.ToGpuContext().Stream;
                var hinPtr = hin.Buffer.Ptr.Reinterpret<float>();
                var xPtr = x.Buffer.Ptr.Reinterpret<float>();
                var prevhPtr = prevh.Buffer.Ptr.Reinterpret<float>();
                var prevcPtr = prevc.Buffer.Ptr.Reinterpret<float>();
                var _ifoaPtr = ifoa1.Buffer.Ptr.Reinterpret<float>();
                var ifoaPtr = ifoa2.Buffer.Ptr.Reinterpret<float>();
                var cPtr = c.Buffer.Ptr.Reinterpret<float>();
                var hPtr = y.Buffer.Ptr.Reinterpret<float>();

                for (var t = 0; t < n; ++t)
                {
                    // stack input
                    stream.For(0, b * xphpb, i =>
                    {
                        var bi = (int)i / xphpb;
                        var _i = (int)i % xphpb;

                        if (_i >= 1 + inputSize) // for hidden
                        {
                            var di = _i - 1 - inputSize;
                            hinPtr[t * b * xphpb + bi * xphpb + _i] = prevhPtr[bi * d + di];
                        }
                        else if (_i >= 1)
                        {
                            var ii = _i - 1;
                            hinPtr[t * b * xphpb + bi * xphpb + _i] = xPtr[t * b * inputSize + bi * inputSize + ii];
                        }
                        else
                        {
                            hinPtr[t * b * xphpb + bi * xphpb + _i] = 1.0f; // bias
                        }
                    });

                    // dot
                    ctx.Assign(ifoa1.Slice(t), Dot(hin.Slice(t).Reshape(b, xphpb), w));

                    // element-wise op
                    stream.For(0, b * d, i =>
                    {
                        var bi = (int)i / d;
                        var di = (int)i % d;

                        var offset1 = t * b * d + bi * d; // for (n, b, d)
                        var offset2 = t * b * 4 * d + bi * 4 * d; // for (n, b, 4*d)
                        var offsetI = offset2;
                        var offsetF = offset2 + d;
                        var offsetO = offset2 + 2 * d;
                        var offsetA = offset2 + 3 * d;

                        var prevct = prevcPtr[bi * d + di];
                        var _it = _ifoaPtr[offsetI + di];
                        var _ft = _ifoaPtr[offsetF + di];
                        var _ot = _ifoaPtr[offsetO + di];
                        var _at = _ifoaPtr[offsetA + di];

                        // non-linearities
                        // a are tanh, others are sigmoid
                        var it = 1.0f / (1.0f + DeviceFunction.Exp(-_it));
                        var ft = 1.0f / (1.0f + DeviceFunction.Exp(-_ft));
                        var ot = 1.0f / (1.0f + DeviceFunction.Exp(-_ot));
                        var at = DeviceFunction.Tanh(_at);

                        // c_t = i_t * a_t + f_t * c_t-1
                        var ct = it * at + ft * prevct;

                        // h_t = o_t * tanh(c_t)
                        var ht = ot * DeviceFunction.Tanh(ct);

                        ifoaPtr[offsetI + di] = it;
                        ifoaPtr[offsetF + di] = ft;
                        ifoaPtr[offsetO + di] = ot;
                        ifoaPtr[offsetA + di] = at;
                        cPtr[offset1 + di] = ct;
                        hPtr[offset1 + di] = ht;
                        prevhPtr[bi * d + di] = ht;
                        prevcPtr[bi * d + di] = ct;
                    });
                }
            }
        }

        public void BackwardOptimized(Executor executor)
        {
            var ctx = executor.Context;
            var one = 1.0.AsScalar<T>();

            var dy = executor.GetGradient(Y); // input
            var w = executor.GetTensor(W);
            var c = executor.GetTensor(C);
            var hin = executor.GetTensor(Hin);
            var ifoa2 = executor.GetTensor(IFOA2);
            var n = (int)dy.Shape[0];
            var b = (int)dy.Shape[1];
            var d = (int)dy.Shape[2];
            var xphpb = (int)w.Shape[0];
            var inputSize = InputSize;

            var cx = executor.GetTensor(CX);
            var hx = executor.GetTensor(HX);
            Util.EnsureTrue(cx.Shape.SequenceEqual(Shape.Create(b, d)));
            Util.EnsureTrue(hx.Shape.SequenceEqual(Shape.Create(b, d)));

            var dc = GetGradient(executor, C);
            var dx = executor.GetGradient(X, Shape.Create(n, b, inputSize));
            var dw = GetGradient(executor, W);
            var difoa1 = GetGradient(executor, IFOA1, ifoa2.Shape);
            var dhin = GetGradient(executor, Hin);
            var dhout = GetGradient(executor, Hout, dy.Shape);
            var dhx = GetGradient(executor, HX);
            var dcx = GetGradient(executor, CX);

            Util.EnsureTrue(ctx.Type == ContextType.Gpu && typeof(T) == typeof(float), "Currently only support gpu and single precision.");

            if (ctx.Type == ContextType.Gpu && typeof(T) == typeof(float))
            {
                var stream = ctx.ToGpuContext().Stream;
                var cxPtr = cx.Buffer.Ptr.Reinterpret<float>();
                var dcxPtr = dcx.Buffer.Ptr.Reinterpret<float>();
                var dhxPtr = dhx.Buffer.Ptr.Reinterpret<float>();
                var cPtr = c.Buffer.Ptr.Reinterpret<float>();
                var dcPtr = dc.Buffer.Ptr.Reinterpret<float>();
                var dhPtr = dhout.Buffer.Ptr.Reinterpret<float>();
                var ifoaPtr = ifoa2.Buffer.Ptr.Reinterpret<float>();
                var _difoaPtr = difoa1.Buffer.Ptr.Reinterpret<float>();
                var dxPtr = dx.Buffer.Ptr.Reinterpret<float>();
                var dhinPtr = dhin.Buffer.Ptr.Reinterpret<float>();
                var dyPtr = dy.Buffer.Ptr.Reinterpret<float>();
                var dwPtr = dw.Buffer.Ptr.Reinterpret<float>();

                // use one kernel to initalize the data
                stream.For(0, Math.Max(n * b * d, xphpb * d), _i =>
                {
                    var i = (int)_i;

                    if (i < n * b * d)
                    {
                        // TODO: dcn and dhn
                        dhPtr[i] = dyPtr[i];
                        dcPtr[i] = 0.0f;
                    }

                    if (i < xphpb * d)
                    {
                        dwPtr[i] = 0.0f;
                    }

                    if (i < b * d)
                    {
                        dhxPtr[i] = 0.0f;
                        dcxPtr[i] = 0.0f;
                    }
                });

                for (var t = n - 1; t >= 0; --t)
                {
                    // c: n, b, d
                    // h: n, b, d
                    // ifoa: n, b, 4*d
                    stream.For(0, b * d, i =>
                    {
                        var bi = (int)i / d;
                        var di = (int)i % d;

                        var offset1 = t * b * d + bi * d; // for (n, b, d)
                        var offset2 = t * b * 4 * d + bi * 4 * d; // for (n, b, 4*d)
                        var offsetI = offset2;
                        var offsetF = offset2 + d;
                        var offsetO = offset2 + 2 * d;
                        var offsetA = offset2 + 3 * d;

                        var ct = cPtr[offset1 + di];
                        var it = ifoaPtr[offsetI + di];
                        var ft = ifoaPtr[offsetF + di];
                        var ot = ifoaPtr[offsetO + di];
                        var at = ifoaPtr[offsetA + di];

                        var dct = dcPtr[offset1 + di];
                        var dht = dhPtr[offset1 + di];

                        var tanhCt = DeviceFunction.Tanh(ct);

                        // do_t = dh_t * tanh(c_t)
                        var dot = dht * tanhCt;

                        // dc_t += dh_t * o_t * (1 - tanh**2(c_t))
                        dct += dht * ot * (1.0f - tanhCt * tanhCt);

                        // df_t = dc_t * c_t-1
                        // dc_t-1 = dc_t * f_t
                        float dft;
                        if (t > 0)
                        {
                            var ctPrev = cPtr[offset1 - b * d + di];
                            dft = dct * ctPrev;
                            dcPtr[offset1 - b * d + di] += ft * dct;
                        }
                        else
                        {
                            var ctPrev = cxPtr[bi * d + di];
                            dft = dct * ctPrev;
                            dcxPtr[bi * d + di] = ft * dct;
                        }
                        // di_t = dc_t * a_t
                        var dit = dct * at;
                        // da_t = dc_t * i_t
                        var dat = dct * it;

                        // backprop activation functions
                        // d^a_t = (1 - a_t * a_t) * da_t (for gradient of tanh)
                        var _dat = (1.0f - at * at) * dat;
                        // others are dv = v*(1-v)*dv (for gradient of sigmoid)
                        var _dit = it * (1.0f - it) * dit;
                        var _dft = ft * (1.0f - ft) * dft;
                        var _dot = ot * (1.0f - ot) * dot;

                        _difoaPtr[offsetI + di] = _dit;
                        _difoaPtr[offsetF + di] = _dft;
                        _difoaPtr[offsetO + di] = _dot;
                        _difoaPtr[offsetA + di] = _dat;
                        dcPtr[offset1 + di] = dct;
                    });

                    // backprop matrix multiply
                    var _difoat = difoa1.Slice(t).Reshape(b, 4 * d);
                    var tmp1 = executor.GetTensor(Temp1, Shape.Create(b, xphpb));
                    ctx.Assign(tmp1, hin.Slice(t).Reshape(b, xphpb));
                    var tmp2 = executor.GetTensor(Temp2, dw.Shape);
                    ctx.Assign(tmp2, Dot(tmp1.T, _difoat));

                    ctx.Assign(dw, dw + tmp2);
                    ctx.Assign(dhin.Slice(t), Dot(_difoat, w.T));

                    // backprop the identity transforms into hin
                    stream.For(0, b * inputSize, i =>
                    {
                        var bi = (int)i / inputSize;
                        var ii = (int)i % inputSize;

                        // write dx of input size
                        // hin: n, b, 1+input+hidden
                        var value = dhinPtr[t * b * xphpb + bi * xphpb + ii + 1];
                        // x: n, b, inputSize
                        dxPtr[t * b * inputSize + bi * inputSize + ii] = value;
                    });

                    // update dh
                    stream.For(0, b * d, i =>
                    {
                        var bi = (int)i / d;
                        var di = (int)i % d;

                        var t0 = t - 1;
                        // dh : n, b, d
                        // dhx : b, d
                        if (t > 0)
                        {
                            dhPtr[t0 * b * d + bi * d + di] += dhinPtr[t * b * xphpb + bi * xphpb + 1 + inputSize + di];
                        }
                        else
                        {
                            dhxPtr[bi * d + di] += dhinPtr[t * b * xphpb + bi * xphpb + 1 + inputSize + di];
                        }
                    });
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

        public Variable<T> Temp2 { get; }
    }
}
