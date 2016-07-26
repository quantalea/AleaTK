using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Runtime.Remoting.Messaging;
using System.Text;
using System.Threading.Tasks;
using Alea;
using Alea.Parallel;
using AleaTK;
using AleaTK.ML;
using AleaTK.ML.Operator;
using csmatio.io;
using csmatio.types;
using NUnit.Framework;
using ICSharpCode.SharpZipLib;
using ICSharpCode.SharpZipLib.Tar;
using static AleaTK.Library;
using static AleaTK.ML.Library;
using Context = AleaTK.Context;

namespace Tutorial.Samples
{
    internal static class TrainPTBUtil
    {
        public static void Iter<T>(this IEnumerable<T> ie, Action<T, int> action)
        {
            var i = 0;
            foreach (var e in ie)
            {
                action(e, i++);
            }
        }

        public static void AreClose(double[] expected, double[] actual, double error)
        {
            if (expected.Length != actual.Length)
            {
                Assert.Fail($"Length doesn't match: {expected.Length} vs {actual.Length}");
            }

            for (var i = 0; i < expected.Length; ++i)
            {
                Assert.That(actual[i], Is.EqualTo(expected[i]).Within(error));
            }
        }

        public static void AreClose(float[] expected, float[] actual, double error)
        {
            if (expected.Length != actual.Length)
            {
                Assert.Fail($"Length doesn't match: {expected.Length} vs {actual.Length}");
            }

            for (var i = 0; i < expected.Length; ++i)
            {
                Assert.That(actual[i], Is.EqualTo(expected[i]).Within(error));
            }
        }

        public static void AreClose(double[,] expected, double[,] actual, double error)
        {
            Assert.AreEqual(expected.GetLength(0), actual.GetLength(0));
            Assert.AreEqual(expected.GetLength(1), actual.GetLength(1));
            for (var row = 0; row < expected.GetLength(0); ++row)
            {
                for (var col = 0; col < expected.GetLength(1); ++col)
                {
                    Assert.That(actual[row, col], Is.EqualTo(expected[row, col]).Within(error));
                }
            }
        }

        public static void AreClose(float[,] expected, float[,] actual, double error)
        {
            Assert.AreEqual(expected.GetLength(0), actual.GetLength(0));
            Assert.AreEqual(expected.GetLength(1), actual.GetLength(1));
            for (var row = 0; row < expected.GetLength(0); ++row)
            {
                for (var col = 0; col < expected.GetLength(1); ++col)
                {
                    Assert.That(actual[row, col], Is.EqualTo(expected[row, col]).Within(error));
                }
            }
        }
    }

    internal static class CSMatIOExtensions
    {
        public static float GetSingle(this MatFileReader reader, string name)
        {
            var marray = reader.GetMLArray(name);
            if (!marray.IsSingle) throw new InvalidCastException("data is not of type float");
            var n = marray.Size;
            var darray = (MLSingle)marray;
            return darray.GetReal(0);
        }

        public static double GetDouble(this MatFileReader reader, string name)
        {
            var marray = reader.GetMLArray(name);
            if (!marray.IsDouble) throw new InvalidCastException("data is not of type double");
            var n = marray.Size;
            var darray = (MLDouble)marray;
            return darray.GetReal(0);
        }

        public static Int64 GetInt64(this MatFileReader reader, string name)
        {
            var marray = reader.GetMLArray(name);
            if (!marray.IsInt64) throw new InvalidCastException("data is not of type Int64");
            var n = marray.Size;
            var darray = (MLInt64)marray;
            return darray.GetReal(0);
        }

        public static UInt64 GetUInt64(this MatFileReader reader, string name)
        {
            var marray = reader.GetMLArray(name);
            if (!marray.IsUInt64) throw new InvalidCastException("data is not of type UInt64");
            var n = marray.Size;
            var darray = (MLUInt64)marray;
            return darray.GetReal(0);
        }

        public static int GetInt(this MatFileReader reader, string name)
        {
            var marray = reader.GetMLArray(name);
            if (!marray.IsInt32) throw new InvalidCastException("data is not of type Int32");
            var n = marray.Size;
            var darray = (MLInt32)marray;
            return darray.GetReal(0);
        }

        public static uint GetUInt(this MatFileReader reader, string name)
        {
            var marray = reader.GetMLArray(name);
            if (!marray.IsUInt32) throw new InvalidCastException("data is not of type UInt32");
            var n = marray.Size;
            var darray = (MLUInt32)marray;
            return darray.GetReal(0);
        }
        public static float[] GetSingleArray(this MatFileReader reader, string name)
        {
            var marray = reader.GetMLArray(name);
            if (!marray.IsSingle) throw new InvalidCastException("data is not of type float");
            var n = marray.Size;
            var darray = (MLSingle)marray;
            var data = new float[n];
            for (var i = 0; i < n; ++i)
                data[i] = darray.GetReal(i);
            return data;
        }
        public static double[] GetDoubleArray(this MatFileReader reader, string name)
        {
            var marray = reader.GetMLArray(name);
            if (!marray.IsDouble) throw new InvalidCastException("data is not of type double");
            var n = marray.Size;
            var darray = (MLDouble)marray;
            var data = new double[n];
            for (var i = 0; i < n; ++i)
                data[i] = darray.GetReal(i);
            return data;
        }

        public static Int64[] GetInt64Array(this MatFileReader reader, string name)
        {
            var marray = reader.GetMLArray(name);
            if (!marray.IsInt64) throw new InvalidCastException("data is not of type Int64");
            var n = marray.Size;
            var darray = (MLInt64)marray;
            var data = new Int64[n];
            for (var i = 0; i < n; ++i)
                data[i] = darray.GetReal(i);
            return data;
        }

        public static UInt64[] GetUInt64Array(this MatFileReader reader, string name)
        {
            var marray = reader.GetMLArray(name);
            if (!marray.IsUInt64) throw new InvalidCastException("data is not of type UInt64");
            var n = marray.Size;
            var darray = (MLUInt64)marray;
            var data = new UInt64[n];
            for (var i = 0; i < n; ++i)
                data[i] = darray.GetReal(i);
            return data;
        }

        public static int[] GetInt32Array(this MatFileReader reader, string name)
        {
            var marray = reader.GetMLArray(name);
            if (!marray.IsInt32) throw new InvalidCastException("data is not of type Int32");
            var n = marray.Size;
            var darray = (MLInt32)marray;
            var data = new int[n];
            for (var i = 0; i < n; ++i)
                data[i] = darray.GetReal(i);
            return data;
        }

        public static uint[] GetUInt32Array(this MatFileReader reader, string name)
        {
            var marray = reader.GetMLArray(name);
            if (!marray.IsUInt32) throw new InvalidCastException("data is not of type UInt32");
            var n = marray.Size;
            var darray = (MLUInt32)marray;
            var data = new uint[n];
            for (var i = 0; i < n; ++i)
                data[i] = darray.GetReal(i);
            return data;
        }
    }

    public static class TrainPTB
    {
        public enum ConfigType
        {
            Small = 0,
            Medium,
            Large
        }

        public const string DataPath = @"Data\PTB\simple-examples\data";
        public const bool Profiling = false;
        public const int TestMaxMaxEpoch = Profiling ? 1 : -1;
        public const int TestHiddenSize = -1;
        //public const ConfigType CfgType = ConfigType.Small;
        public const ConfigType CfgType = ConfigType.Medium;
        //public const ConfigType CfgType = ConfigType.Large;

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
                    Parameter(RandomNormal<T>(Shape.Create(InputSize + HiddenSize + 1, 4 * HiddenSize)) /
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
                    ctx.Assign(w.Slice(0, Range(HiddenSize, 2 * HiddenSize)),
                        Fill(Shape.Create(1, HiddenSize), ScalarOps.Conv<T>(ForgetBiasInit)));
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

            public const bool UseOpt = true;

            public override void Forward(Executor executor)
            {
                if (UseOpt)
                {
                    Forward2(executor);
                }
                else
                {
                    Forward1(executor);
                }
            }

            public override void Backward(Executor executor)
            {
                if (UseOpt)
                {
                    Backward2(executor);
                }
                else
                {
                    Backward1(executor);
                }
            }

            public void Forward1(Executor executor)
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

            public void Backward1(Executor executor)
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

            public void Forward2(Executor executor)
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

            public void Backward2(Executor executor)
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

        [Test]
        public static void LSTMvsPython()
        {
            var mfr = new MatFileReader(@"lstm_small.mat");

            var inputSize = mfr.GetInt("InputSize");
            var seqLength = mfr.GetInt("SeqLength");
            var hiddenSize = mfr.GetInt("HiddenSize");
            var batchSize = mfr.GetInt("BatchSize");

            var x = Variable<float>(PartialShape.Create(seqLength, batchSize, inputSize));
            var lstm = new LSTM<float>(x, hiddenSize);

            var ctx = Context.GpuContext(0);
            var exe = new Executor(ctx, lstm.Y);

            exe.Initalize();

            var h0 = mfr.GetDoubleArray("h0").Select(n => (float)n).ToArray();
            var c0 = mfr.GetDoubleArray("c0").Select(n => (float)n).ToArray();
            exe.AssignTensor(lstm.CX, c0.AsTensor(Shape.Create(batchSize, hiddenSize)));
            exe.AssignTensor(lstm.HX, h0.AsTensor(Shape.Create(batchSize, hiddenSize)));

            var input = mfr.GetDoubleArray("X").Select(n => (float)n).ToArray();
            exe.AssignTensor(x, input.AsTensor(Shape.Create(seqLength, batchSize, inputSize)));

            var w = mfr.GetDoubleArray("W").Select(n => (float)n).ToArray();
            w.AsTensor(Shape.Create(inputSize + hiddenSize + 1, 4 * hiddenSize)).Print();
            exe.AssignTensor(lstm.W, w.AsTensor(Shape.Create(inputSize + hiddenSize + 1, 4 * hiddenSize)));

            exe.Forward();

            var H = mfr.GetDoubleArray("H").Select(n => (float)n).ToArray();
            H.AsTensor(Shape.Create(seqLength * batchSize, hiddenSize)).Print();

            var myH = exe.GetTensor(lstm.Y).ToArray();
            myH.AsTensor(Shape.Create(seqLength * batchSize, hiddenSize)).Print();

            TrainPTBUtil.AreClose(H, myH, 1e-6);

            var CN = mfr.GetDoubleArray("cn").Select(n => (float)n).ToArray();
            CN.AsTensor(Shape.Create(batchSize, hiddenSize)).Print();

            var myCN = exe.GetTensor(lstm.CY).ToArray();
            myCN.AsTensor(Shape.Create(batchSize, hiddenSize)).Print();

            TrainPTBUtil.AreClose(CN, myCN, 1e-6);

            var HN = mfr.GetDoubleArray("hn").Select(n => (float)n).ToArray();
            HN.AsTensor(Shape.Create(batchSize, hiddenSize)).Print();

            var myHN = exe.GetTensor(lstm.HY).ToArray();
            myHN.AsTensor(Shape.Create(batchSize, hiddenSize)).Print();

            TrainPTBUtil.AreClose(HN, myHN, 1e-6);

            var dH = mfr.GetDoubleArray("dH").Select(n => (float)n).ToArray();
            exe.AssignGradientDirectly(lstm.Y, dH.AsTensor(Shape.Create(seqLength, batchSize, hiddenSize)));

            exe.Backward();

            var dX = mfr.GetDoubleArray("dX").Select(n => (float)n).ToArray();
            dX.AsTensor(Shape.Create(seqLength * batchSize, inputSize)).Print();

            var dXmy = exe.GetGradient(lstm.X).ToArray();
            dXmy.AsTensor(Shape.Create(seqLength * batchSize, inputSize)).Print();
            TrainPTBUtil.AreClose(dX, dXmy, 1e-6);

            var dW = mfr.GetDoubleArray("dW").Select(n => (float)n).ToArray();
            dW.AsTensor(Shape.Create(inputSize + hiddenSize + 1, 4 * hiddenSize)).Print();

            var dWmy = exe.GetGradient(lstm.W).ToArray();
            dWmy.AsTensor(Shape.Create(lstm.W.Shape.AsArray)).Print();
            TrainPTBUtil.AreClose(dW, dWmy, 1e-6);

            var dc0 = mfr.GetDoubleArray("dc0").Select(n => (float)n).ToArray();
            dc0.AsTensor(Shape.Create(batchSize, hiddenSize)).Print();

            var dc0my = exe.GetGradient(lstm.CX).ToArray();
            dc0my.AsTensor(Shape.Create(batchSize, hiddenSize)).Print();
            TrainPTBUtil.AreClose(dc0, dc0my, 1e-6);

            var dh0 = mfr.GetDoubleArray("dh0").Select(n => (float)n).ToArray();
            dh0.AsTensor(Shape.Create(batchSize, hiddenSize)).Print();

            var dh0my = exe.GetGradient(lstm.HX).ToArray();
            dh0my.AsTensor(Shape.Create(batchSize, hiddenSize)).Print();
            TrainPTBUtil.AreClose(dh0, dh0my, 1e-6);

            ctx.ToGpuContext().Stream.Synchronize();
        }

        [Test]
        public static void LSTMvsCUDNN()
        {
            var ctx = Context.GpuContext(0);
            var inputSize = 5;
            var seqLength = 3;
            var batchSize = 2;
            var hiddenSize = 4;
            //var inputSize = 50;
            //var seqLength = 30;
            //var batchSize = 20;
            //var hiddenSize = 40;
            var error = 1e-5;

            var data =
                Context.CpuContext.Eval((2.0f.AsScalar() *
                                         RandomUniform<float>(Shape.Create(seqLength, batchSize, inputSize)) -
                                         1.0f.AsScalar())).ToArray3D();
            //data.AsTensor(Shape.Create(seqLength*batchSize, inputSize)).Print();

            //var h0 = Context.CpuContext.Eval(Fill(Shape.Create(batchSize, hiddenSize), 0.0f)).ToArray2D();
            //var c0 = Context.CpuContext.Eval(Fill(Shape.Create(batchSize, hiddenSize), 0.0f)).ToArray2D();
            var h0 = Context.CpuContext.Eval(RandomNormal<float>(Shape.Create(batchSize, hiddenSize))).ToArray2D();
            var c0 = Context.CpuContext.Eval(RandomNormal<float>(Shape.Create(batchSize, hiddenSize))).ToArray2D();

            var dy =
                Context.CpuContext.Eval((2.0f.AsScalar() *
                                         RandomUniform<float>(Shape.Create(seqLength, batchSize, hiddenSize)) -
                                         1.0f.AsScalar())).ToArray3D();
            //dy.AsTensor(Shape.Create(seqLength * batchSize, hiddenSize)).Print();

            var wi = 0.5f;
            var wf = 0.4f;
            var wo = 0.3f;
            var wa = 0.2f;
            var ui = 0.5f;
            var uf = 0.4f;
            var uo = 0.3f;
            var ua = 0.1f;
            var bi = 0.5f;
            var bf = 0.4f;
            var bo = 0.3f;
            var ba = 0.2f;

            float[,,] y1, y2, dx1, dx2;
            float[,] cy1, cy2, hy1, hy2;
            float[,] dcx1, dcx2, dhx1, dhx2;
            float[,] dw1, dw2;
            {
                // calc with cuDNN
                var x = Variable<float>(PartialShape.Create(seqLength, batchSize, inputSize));
                var lstm = new RNN<float>(x, 1, hiddenSize, dropout: 0.0);
                var exe = new Executor(ctx, lstm.Y);
                exe.Initalize();

                // set input
                exe.AssignTensor(lstm.X, data.AsTensor());

                // set states
                exe.AssignTensor(lstm.CX, c0.AsTensor(Shape.Create(1, batchSize, hiddenSize)));
                exe.AssignTensor(lstm.HX, h0.AsTensor(Shape.Create(1, batchSize, hiddenSize)));

                // set weigths
                // cuDNN matrices order: IFAO
                var w = exe.GetTensor(lstm.W).Reshape(inputSize * 4 + hiddenSize * 4 + 2 * 4, hiddenSize);
                var offset = 0;
                // Wi
                ctx.Assign(w.Slice(Range(offset, offset + inputSize)), Fill(Shape.Create(inputSize, hiddenSize), wi));
                offset += inputSize;
                // Wf
                ctx.Assign(w.Slice(Range(offset, offset + inputSize)), Fill(Shape.Create(inputSize, hiddenSize), wf));
                offset += inputSize;
                // Wa
                ctx.Assign(w.Slice(Range(offset, offset + inputSize)), Fill(Shape.Create(inputSize, hiddenSize), wa));
                offset += inputSize;
                // Wo
                ctx.Assign(w.Slice(Range(offset, offset + inputSize)), Fill(Shape.Create(inputSize, hiddenSize), wo));
                offset += inputSize;
                // Ui
                ctx.Assign(w.Slice(Range(offset, offset + hiddenSize)), Fill(Shape.Create(hiddenSize, hiddenSize), ui));
                offset += hiddenSize;
                // Uf
                ctx.Assign(w.Slice(Range(offset, offset + hiddenSize)), Fill(Shape.Create(hiddenSize, hiddenSize), uf));
                offset += hiddenSize;
                // Ua
                ctx.Assign(w.Slice(Range(offset, offset + hiddenSize)), Fill(Shape.Create(hiddenSize, hiddenSize), ua));
                offset += hiddenSize;
                // Uo
                ctx.Assign(w.Slice(Range(offset, offset + hiddenSize)), Fill(Shape.Create(hiddenSize, hiddenSize), uo));
                offset += hiddenSize;
                // Bi
                ctx.Assign(w.Slice(offset), Fill(Shape.Create(1, hiddenSize), bi));
                offset++;
                // Bf
                ctx.Assign(w.Slice(offset), Fill(Shape.Create(1, hiddenSize), bf));
                offset++;
                // Ba
                ctx.Assign(w.Slice(offset), Fill(Shape.Create(1, hiddenSize), ba));
                offset++;
                // Bo
                ctx.Assign(w.Slice(offset), Fill(Shape.Create(1, hiddenSize), bo));

                exe.Forward();

                y1 = exe.GetTensor(lstm.Y).ToArray3D();
                cy1 = exe.GetTensor(lstm.CY).Reshape(batchSize, hiddenSize).ToArray2D();
                hy1 = exe.GetTensor(lstm.HY).Reshape(batchSize, hiddenSize).ToArray2D();

                exe.AssignGradientDirectly(lstm.Y, dy.AsTensor());

                exe.Backward();

                dx1 = exe.GetGradient(lstm.X).ToArray3D();
                dcx1 = exe.GetGradient(lstm.CX).Reshape(batchSize, hiddenSize).ToArray2D();
                dhx1 = exe.GetGradient(lstm.HX).Reshape(batchSize, hiddenSize).ToArray2D();

                // we make dw follow the shape as (1 + inputSize + hiddenSize, 4*hiddenSize)
                // also cuDNN is fortran order, so we need transpose it
                var dwCUDNN = exe.GetGradient(lstm.W).ToArray().AsTensor();
                dw1 = new float[1 + inputSize + hiddenSize, 4 * hiddenSize];
                var dw1Tensor = Reference<float>(dw1);
                var cpu = Context.CpuContext;
                offset = 0;
                // cuDNN order: IFAO , also need transpose, cuDNN stores data in fortran order.
                // Wi
                cpu.Assign(dw1Tensor.Slice(Range(1, inputSize + 1), Range(0, hiddenSize)), dwCUDNN.Slice(Range(offset, offset + inputSize * hiddenSize)).Reshape(hiddenSize, inputSize).T);
                offset += inputSize * hiddenSize;
                // Wf
                cpu.Assign(dw1Tensor.Slice(Range(1, inputSize + 1), Range(hiddenSize, 2 * hiddenSize)), dwCUDNN.Slice(Range(offset, offset + inputSize * hiddenSize)).Reshape(hiddenSize, inputSize).T);
                offset += inputSize * hiddenSize;
                // Wa
                cpu.Assign(dw1Tensor.Slice(Range(1, inputSize + 1), Range(3 * hiddenSize, 4 * hiddenSize)), dwCUDNN.Slice(Range(offset, offset + inputSize * hiddenSize)).Reshape(hiddenSize, inputSize).T);
                offset += inputSize * hiddenSize;
                // Wo
                cpu.Assign(dw1Tensor.Slice(Range(1, inputSize + 1), Range(2 * hiddenSize, 3 * hiddenSize)), dwCUDNN.Slice(Range(offset, offset + inputSize * hiddenSize)).Reshape(hiddenSize, inputSize).T);
                offset += inputSize * hiddenSize;
                // Ui
                cpu.Assign(dw1Tensor.Slice(Range(inputSize + 1, -1), Range(0, hiddenSize)), dwCUDNN.Slice(Range(offset, offset + hiddenSize * hiddenSize)).Reshape(hiddenSize, hiddenSize).T);
                offset += hiddenSize * hiddenSize;
                // Uf
                cpu.Assign(dw1Tensor.Slice(Range(inputSize + 1, -1), Range(hiddenSize, 2 * hiddenSize)), dwCUDNN.Slice(Range(offset, offset + hiddenSize * hiddenSize)).Reshape(hiddenSize, hiddenSize).T);
                offset += hiddenSize * hiddenSize;
                // Ua
                cpu.Assign(dw1Tensor.Slice(Range(inputSize + 1, -1), Range(3 * hiddenSize, 4 * hiddenSize)), dwCUDNN.Slice(Range(offset, offset + hiddenSize * hiddenSize)).Reshape(hiddenSize, hiddenSize).T);
                offset += hiddenSize * hiddenSize;
                // Uo
                cpu.Assign(dw1Tensor.Slice(Range(inputSize + 1, -1), Range(2 * hiddenSize, 3 * hiddenSize)), dwCUDNN.Slice(Range(offset, offset + hiddenSize * hiddenSize)).Reshape(hiddenSize, hiddenSize).T);
                offset += hiddenSize * hiddenSize;
                // Bi
                cpu.Assign(dw1Tensor.Slice(0, Range(0, hiddenSize)), dwCUDNN.Slice(Range(offset, offset + hiddenSize)).Reshape(hiddenSize, 1).T);
                offset += hiddenSize;
                // Bf
                cpu.Assign(dw1Tensor.Slice(0, Range(hiddenSize, 2 * hiddenSize)), dwCUDNN.Slice(Range(offset, offset + hiddenSize)).Reshape(hiddenSize, 1).T);
                offset += hiddenSize;
                // Ba
                cpu.Assign(dw1Tensor.Slice(0, Range(3 * hiddenSize, 4 * hiddenSize)), dwCUDNN.Slice(Range(offset, offset + hiddenSize)).Reshape(hiddenSize, 1).T);
                offset += hiddenSize;
                // Bo
                cpu.Assign(dw1Tensor.Slice(0, Range(2 * hiddenSize, 3 * hiddenSize)), dwCUDNN.Slice(Range(offset, offset + hiddenSize)).Reshape(hiddenSize, 1).T);
            }

            {
                // calc with lstm
                var x = Variable<float>(PartialShape.Create(seqLength, batchSize, inputSize));
                var lstm = new LSTM<float>(x, hiddenSize, forgetBiasInit: 0.0);
                var exe = new Executor(ctx, lstm.Y);
                exe.Initalize();

                // set input
                exe.AssignTensor(lstm.X, data.AsTensor());

                // set states
                exe.AssignTensor(lstm.CX, c0.AsTensor());
                exe.AssignTensor(lstm.HX, h0.AsTensor());

                // set weights
                var w = exe.GetTensor(lstm.W);
                // Wi
                ctx.Assign(w.Slice(Range(1, inputSize + 1), Range(0, hiddenSize)), Fill(Shape.Create(inputSize, hiddenSize), wi));
                // Wf
                ctx.Assign(w.Slice(Range(1, inputSize + 1), Range(hiddenSize, 2 * hiddenSize)), Fill(Shape.Create(inputSize, hiddenSize), wf));
                // Wo
                ctx.Assign(w.Slice(Range(1, inputSize + 1), Range(2 * hiddenSize, 3 * hiddenSize)), Fill(Shape.Create(inputSize, hiddenSize), wo));
                // Wa
                ctx.Assign(w.Slice(Range(1, inputSize + 1), Range(3 * hiddenSize, 4 * hiddenSize)), Fill(Shape.Create(inputSize, hiddenSize), wa));
                // Ui
                ctx.Assign(w.Slice(Range(inputSize + 1, -1), Range(0, hiddenSize)), Fill(Shape.Create(hiddenSize, hiddenSize), ui));
                // Uf
                ctx.Assign(w.Slice(Range(inputSize + 1, -1), Range(hiddenSize, 2 * hiddenSize)), Fill(Shape.Create(hiddenSize, hiddenSize), uf));
                // Uo
                ctx.Assign(w.Slice(Range(inputSize + 1, -1), Range(2 * hiddenSize, 3 * hiddenSize)), Fill(Shape.Create(hiddenSize, hiddenSize), uo));
                // Ua
                ctx.Assign(w.Slice(Range(inputSize + 1, -1), Range(3 * hiddenSize, 4 * hiddenSize)), Fill(Shape.Create(hiddenSize, hiddenSize), ua));
                // Bi
                ctx.Assign(w.Slice(0, Range(0, hiddenSize)), Fill(Shape.Create(1, hiddenSize), bi));
                // Bf
                ctx.Assign(w.Slice(0, Range(hiddenSize, 2 * hiddenSize)), Fill(Shape.Create(1, hiddenSize), bf));
                // Bo
                ctx.Assign(w.Slice(0, Range(2 * hiddenSize, 3 * hiddenSize)), Fill(Shape.Create(1, hiddenSize), bo));
                // Ba
                ctx.Assign(w.Slice(0, Range(3 * hiddenSize, 4 * hiddenSize)), Fill(Shape.Create(1, hiddenSize), ba));

                exe.Forward();

                y2 = exe.GetTensor(lstm.Y).ToArray3D();
                cy2 = exe.GetTensor(lstm.CY).ToArray2D();
                hy2 = exe.GetTensor(lstm.HY).ToArray2D();

                exe.AssignGradientDirectly(lstm.Y, dy.AsTensor());

                exe.Backward();

                dx2 = exe.GetGradient(lstm.X).ToArray3D();
                dcx2 = exe.GetGradient(lstm.CX).Reshape(batchSize, hiddenSize).ToArray2D();
                dhx2 = exe.GetGradient(lstm.HX).Reshape(batchSize, hiddenSize).ToArray2D();
                dw2 = exe.GetGradient(lstm.W).ToArray2D();
            }

            //y1.AsTensor(Shape.Create(seqLength*batchSize, hiddenSize)).Print();
            //y2.AsTensor(Shape.Create(seqLength*batchSize, hiddenSize)).Print();
            TrainPTBUtil.AreClose(y1.AsTensor().ToArray(), y2.AsTensor().ToArray(), error);

            //cy1.AsTensor().Print();
            //cy2.AsTensor().Print();
            TrainPTBUtil.AreClose(cy1, cy2, error);

            //hy1.AsTensor().Print();
            //hy2.AsTensor().Print();
            TrainPTBUtil.AreClose(hy1, hy2, error);

            //dx1.AsTensor(Shape.Create(seqLength * batchSize, inputSize)).Print();
            //dx2.AsTensor(Shape.Create(seqLength * batchSize, inputSize)).Print();
            TrainPTBUtil.AreClose(dx1.AsTensor().ToArray(), dx2.AsTensor().ToArray(), error);

            //dcx1.AsTensor().Print();
            //dcx2.AsTensor().Print();
            TrainPTBUtil.AreClose(dcx1, dcx2, error);

            //dhx1.AsTensor().Print();
            //dhx2.AsTensor().Print();
            TrainPTBUtil.AreClose(dhx1, dhx2, error);

            //dw1.AsTensor().Print();
            //dw2.AsTensor().Print();
            TrainPTBUtil.AreClose(dw1, dw2, error);
        }

        public class Config
        {
            public double InitScale;
            public double LearningRate;
            public double MaxGradNorm;
            public int NumLayers;
            public int NumSteps;
            public int HiddenSize;
            public int MaxEpoch; // learning rate start to reduce after this epoch
            public int MaxMaxEpoch; // epoches to run
            public double KeepProb;
            public double LrDecay;
            public int BatchSize;
            public int VocabSize;

            public static Config Small(int batchSize = 20, int numSteps = 20, double keepProb = 1.0)
            {
                return new Config
                {
                    InitScale = 0.1,
                    LearningRate = 1.0,
                    MaxGradNorm = 5.0,
                    NumLayers = 2,
                    NumSteps = numSteps,
                    HiddenSize = TestHiddenSize > 0 ? TestHiddenSize : 200,
                    MaxEpoch = 4,
                    MaxMaxEpoch = TestMaxMaxEpoch > 0 ? TestMaxMaxEpoch : 13,
                    KeepProb = keepProb,
                    LrDecay = 0.5,
                    BatchSize = batchSize,
                    VocabSize = 10000
                };
            }

            public static Config Medium(int batchSize = 20, int numSteps = 35, double keepProb = 0.5)
            {
                return new Config
                {
                    InitScale = 0.05,
                    LearningRate = 1.0,
                    MaxGradNorm = 5.0,
                    NumLayers = 2,
                    NumSteps = numSteps,
                    HiddenSize = TestHiddenSize > 0 ? TestHiddenSize : 650,
                    MaxEpoch = 6,
                    MaxMaxEpoch = TestMaxMaxEpoch > 0 ? TestMaxMaxEpoch : 39,
                    KeepProb = keepProb,
                    LrDecay = 0.8,
                    BatchSize = batchSize,
                    VocabSize = 10000
                };
            }

            public static Config Large(int batchSize = 20, int numSteps = 35, double keepProb = 0.35)
            {
                return new Config
                {
                    InitScale = 0.04,
                    LearningRate = 1.0,
                    MaxGradNorm = 10.0,
                    NumLayers = 2,
                    NumSteps = numSteps,
                    HiddenSize = TestHiddenSize > 0 ? TestHiddenSize : 1500,
                    MaxEpoch = 14,
                    MaxMaxEpoch = TestMaxMaxEpoch > 0 ? TestMaxMaxEpoch : 55,
                    KeepProb = keepProb,
                    LrDecay = 1.0/1.15,
                    BatchSize = batchSize,
                    VocabSize = 10000
                };
            }
        }

        [Test, Ignore("This is just developing test.")]
        public static void TestEnsureDataFile()
        {
            Data.EnsureDataFile();
        }

        public class Data
        {
            private static void Decompress(string src, string dst)
            {
                using (var originalFileStream = File.OpenRead(src))
                using (var decompressedFileStream = File.Create(dst))
                using (var decompressionStream = new GZipStream(originalFileStream, CompressionMode.Decompress))
                {
                    decompressionStream.CopyTo(decompressedFileStream);
                }
            }

            public static void EnsureDataFile()
            {
                const string doneFileName = @"Data\PTB.done";
                const string url = @"http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz";

                if (!Directory.Exists("Data"))
                {
                    Directory.CreateDirectory("Data");
                }

                if (!File.Exists(doneFileName))
                {
                    using (var client = new WebClient())
                    {
                        Console.WriteLine($"Downloading {url} ...");
                        client.DownloadFile(url, @"Data\PTB.tgz");
                    }

                    Decompress(@"Data\PTB.tgz", @"Data\PTB.tar");

                    using (var tarFile = File.OpenRead(@"Data\PTB.tar"))
                    using (var tarArchive = TarArchive.CreateInputTarArchive(tarFile))
                    {
                        tarArchive.ExtractContents(@"Data\PTB");
                    }

                    using (var doneFile = File.CreateText(doneFileName))
                    {
                        doneFile.WriteLine($"{DateTime.Now}");
                    }
                }
            }

            public static List<string> ReadWords(string path)
            {
                var totalWords = new List<string>();
                using (var file = File.Open(path, FileMode.Open))
                using (var reader = new StreamReader(file, Encoding.UTF8))
                {
                    while (!reader.EndOfStream)
                    {
                        var line = reader.ReadLine();
                        var words = line?.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                        if (!(words?.Length > 0)) continue;
                        totalWords.AddRange(words);
                        totalWords.Add("<eos>");
                    }
                }
                return totalWords;
            }

            public static void BuildVocab(string path, out Dictionary<string, int> word2id, out Dictionary<int, string> id2word)
            {
                var data = ReadWords(path).Distinct().ToList();
                data.Sort();
                word2id = new Dictionary<string, int>();
                id2word = new Dictionary<int, string>();
                var id = 0;
                //foreach (var word in data)
                //{
                //    Console.WriteLine(word);
                //}
                foreach (var word in data)
                {
                    word2id.Add(word, id);
                    id2word.Add(id, word);
                    id++;
                }
            }

            public readonly Dictionary<string, int> WordToIdDict;

            public readonly Dictionary<int, string> IdToWordDict;

            public readonly int[] TrainData;

            public readonly int[] ValidData;

            public readonly int[] TestData;

            public int WordToId(string word)
            {
                return WordToIdDict.ContainsKey(word) ? WordToIdDict[word] : WordToIdDict["<unk>"];
            }

            public string IdToWord(int id)
            {
                return IdToWordDict[id];
            }

            public Data(string dataPath)
            {
                EnsureDataFile();

                var TrainPath = Path.Combine(dataPath, "ptb.train.txt");
                var ValidPath = Path.Combine(dataPath, "ptb.valid.txt");
                var TestPath = Path.Combine(dataPath, "ptb.test.txt");

                BuildVocab(TrainPath, out WordToIdDict, out IdToWordDict);

                TrainData = ReadWords(TrainPath).Select(WordToId).ToArray();
                ValidData = ReadWords(ValidPath).Select(WordToId).ToArray();
                TestData = ReadWords(TestPath).Select(WordToId).ToArray();
            }

            public class Batch
            {
                public int[,] Inputs { get; set; }
                public int[,] Targets { get; set; }
            }

            public static IEnumerable<Batch> Iterator(int[] rawData, int numSteps, int batchSize)
            {
                var dataLen = rawData.Length;
                var batchLen = dataLen / batchSize;
                var data = new int[batchSize, batchLen];
                for (var i = 0; i < batchSize; ++i)
                {
                    for (var j = 0; j < batchLen; ++j)
                    {
                        data[i, j] = rawData[batchLen * i + j];
                    }
                }

                var epochSize = (batchLen - 1) / numSteps;

                Util.EnsureTrue(epochSize != 0);

                for (var i = 0; i < epochSize; ++i)
                {
                    var x = new int[numSteps, batchSize];
                    var y = new int[numSteps, batchSize];

                    for (var t = 0; t < numSteps; ++t)
                    {
                        for (var j = 0; j < batchSize; ++j)
                        {
                            x[t, j] = data[j, numSteps*i + t];
                            y[t, j] = data[j, numSteps*i + t + 1];
                        }
                    }

                    yield return new Batch { Inputs = x, Targets = y };
                }
            }
        }

        public class IndexAndProb : IComparable
        {
            public int Index;
            public double Prob;

            public int CompareTo(object obj)
            {
                var o = (IndexAndProb)obj;
                if (Prob == o.Prob) return 0;
                return Prob > o.Prob ? -1 : 1;
            }

            public override string ToString()
            {
                return $"({Index}:{Prob:F2})";
            }
        }

        // This model uses our LSTM implementation
        public class Model1
        {
            public Model1(Context ctx, Config cfg, bool isTraining = true)
            {
                Config = cfg;
                IsTraining = isTraining;

                Inputs = Variable<int>(PartialShape.Create(cfg.NumSteps, cfg.BatchSize));
                Targets = Variable<int>(PartialShape.Create(cfg.NumSteps, cfg.BatchSize));

                // embedding, possible dropout
                Embedding = new Embedding<float>(Inputs, cfg.VocabSize, cfg.HiddenSize, initScale: cfg.InitScale);
                EmbeddedOutput = Embedding.Output;
                if (isTraining && cfg.KeepProb < 1.0)
                {
                    var dropout = new Dropout<float>(EmbeddedOutput, dropoutProb: 1.0 - cfg.KeepProb);
                    EmbeddedOutput = dropout.Output;
                }

                // rnn layer, possible dropout for each lstm layer output
                RNN = new LSTM<float>[cfg.NumLayers];
                for (var i = 0; i < cfg.NumLayers; ++i)
                {
                    var lstm = new LSTM<float>(i == 0 ? EmbeddedOutput : RNNOutput, cfg.HiddenSize, forgetBiasInit: 0.0);
                    RNN[i] = lstm;
                    RNNOutput = lstm.Y;
                    if (isTraining && cfg.KeepProb < 1.0)
                    {
                        var dropout = new Dropout<float>(RNNOutput, dropoutProb: 1.0 - cfg.KeepProb);
                        RNNOutput = dropout.Output;
                    }
                }

                FC =
                    new FullyConnected<float>(RNNOutput.Reshape(RNNOutput.Shape[0]*RNNOutput.Shape[1],
                        RNNOutput.Shape[2]), cfg.VocabSize);

                Loss = new SoftmaxCrossEntropySparse<float>(FC.Output,
                    Targets.Reshape(Targets.Shape[0] * Targets.Shape[1]));

                Optimizer = new GradientDescentOptimizer(ctx, Loss.Loss, cfg.LearningRate,
                    new GlobalNormGradientClipper(cfg.MaxGradNorm));

                // warmup (for JIT, and better timing measure)
                Optimizer.Initalize();
                ResetStates();
                Optimizer.AssignTensor(Inputs, Fill(Shape.Create(Inputs.Shape.AsArray), 0));
                Optimizer.AssignTensor(Targets, Fill(Shape.Create(Targets.Shape.AsArray), 0));
                Optimizer.Forward();
                if (isTraining)
                {
                    // TODO
                    Optimizer.Backward();
                }

                // now reset states
                Optimizer.Initalize();
                ResetStates();
            }

            public void CopyWeightsFrom(Model1 o)
            {
                Optimizer.AssignTensor(Embedding.Weights, o.Optimizer.GetTensor(o.Embedding.Weights));
                for (var i = 0; i < Config.NumLayers; ++i)
                {
                    Optimizer.AssignTensor(RNN[i].W, o.Optimizer.GetTensor(o.RNN[i].W));
                }
                Optimizer.AssignTensor(FC.Weights, o.Optimizer.GetTensor(o.FC.Weights));
                Optimizer.AssignTensor(FC.Bias, o.Optimizer.GetTensor(o.FC.Bias));
            }

            public void ResetStates()
            {
                for (var i = 0; i < Config.NumLayers; ++i)
                {
                    var lstm = RNN[i];
                    var shape = Shape.Create(Config.BatchSize, lstm.HiddenSize);
                    Optimizer.AssignTensor(lstm.CX, Fill(shape, 0.0f));
                    Optimizer.AssignTensor(lstm.HX, Fill(shape, 0.0f));
                }
            }

            public void CopyStates()
            {
                for (var i = 0; i < Config.NumLayers; ++i)
                {
                    var lstm = RNN[i];
                    Optimizer.AssignTensor(lstm.CX, Optimizer.GetTensor(lstm.CY));
                    Optimizer.AssignTensor(lstm.HX, Optimizer.GetTensor(lstm.HY));
                }
            }

            public double RunEpoch(int[] data, double learningRate = 1.0, bool verbose = false)
            {
                var cfg = Config;
                var isTraining = IsTraining;
                var epochSize = (data.Length / cfg.BatchSize - 1) / cfg.NumSteps;
                var time = Stopwatch.StartNew();
                var costs = 0.0;
                var iters = 0;
                var step = 0;
                var firstBatch = true;

                foreach (var batch in Data.Iterator(data, cfg.NumSteps, cfg.BatchSize))
                {
                    Optimizer.AssignTensor(Inputs, batch.Inputs.AsTensor());
                    Optimizer.AssignTensor(Targets, batch.Targets.AsTensor());

                    if (firstBatch)
                    {
                        ResetStates();
                        firstBatch = false;
                    }
                    else
                    {
                        CopyStates();
                    }

                    Optimizer.Forward();

                    if (isTraining)
                    {
                        Optimizer.Backward();
                        Optimizer.Optimize(learningRate);
                    }

                    var loss = Optimizer.GetTensor(Loss.Loss).ToScalar();
                    var cost = loss / cfg.BatchSize;
                    costs += cost;
                    iters += cfg.NumSteps;

                    if (Profiling || (verbose && (step % (epochSize / 10) == 10)))
                    {
                        var perplexity = Math.Exp(costs / iters);
                        var wps = (iters * cfg.BatchSize) / (time.Elapsed.TotalMilliseconds / 1000.0);

                        Console.WriteLine($"{step:D4}: {step * 1.0 / epochSize:F3} perplexity: {perplexity:F3} speed:{wps:F0} wps cost: {cost:F3}");
                    }

                    if (Profiling && step > 5) break;

                    step++;
                }
                return Math.Exp(costs / iters);
            }

            public Config Config { get; }

            public bool IsTraining { get; }

            public Variable<int> Inputs { get; }

            public Variable<int> Targets { get; }

            public Embedding<float> Embedding { get; }

            public Variable<float> EmbeddedOutput { get; }

            public LSTM<float>[] RNN { get; } 

            public Variable<float> RNNOutput { get; }

            public FullyConnected<float> FC { get; }

            public SoftmaxCrossEntropySparse<float> Loss { get; }

            public GradientDescentOptimizer Optimizer { get; }
        }

        [Test, Ignore("To long to run, please explicitly run it.")]
        public static void Run1()
        {
            Run1(false, CfgType);
        }

        public static void Run1(bool isConsole, ConfigType cfgType)
        {
            Console.WriteLine($"Scratch Version, Config: {cfgType}");

            var ptb = new Data(DataPath);
            var ctx = Context.GpuContext(0);

            Config cfg, cfgValid, cfgTest, cfgInteractive;

            switch (cfgType)
            {
                case ConfigType.Small:
                    cfg = Config.Small(batchSize: 20);
                    cfgValid = Config.Small(batchSize: 20);
                    cfgTest = Config.Small(batchSize: 1, numSteps: 1);
                    cfgInteractive = Config.Small(batchSize: 1, numSteps: 10);
                    break;
                case ConfigType.Medium:
                    cfg = Config.Medium(batchSize: 20);
                    cfgValid = Config.Medium(batchSize: 20);
                    cfgTest = Config.Medium(batchSize: 1, numSteps: 1);
                    cfgInteractive = Config.Medium(batchSize: 1, numSteps: 10);
                    break;
                case ConfigType.Large:
                    cfg = Config.Large(batchSize: 20);
                    cfgValid = Config.Large(batchSize: 20);
                    cfgTest = Config.Large(batchSize: 1, numSteps: 1);
                    cfgInteractive = Config.Large(batchSize: 1, numSteps: 10);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(cfgType), cfgType, null);
            }

            Assert.AreEqual(ptb.WordToIdDict.Count, cfg.VocabSize);
            Assert.AreEqual(ptb.WordToIdDict.Count, cfgValid.VocabSize);
            Assert.AreEqual(ptb.WordToIdDict.Count, cfgTest.VocabSize);
            Assert.AreEqual(ptb.WordToIdDict.Count, cfgInteractive.VocabSize);

            var model = new Model1(ctx, cfg, isTraining: true);
            var modelValid = new Model1(ctx, cfgValid, isTraining: false);
            var modelTest = new Model1(ctx, cfgTest, isTraining: false);
            var modelInteractive = new Model1(ctx, cfgInteractive, isTraining: false);

            for (var i = 0; i < cfg.MaxMaxEpoch; ++i)
            {
                var lrDecay = Math.Pow(cfg.LrDecay, Math.Max(i - cfg.MaxEpoch, 0.0));
                var learningRate = cfg.LearningRate*lrDecay;

                Console.WriteLine($"Epoch: {i + 1} Learning rate: {learningRate:F3}");
                var trainPerplexity = model.RunEpoch(ptb.TrainData, learningRate: learningRate, verbose: true);
                Console.WriteLine($"Epoch: {i + 1} Train Perplexity: {trainPerplexity:F3}");

                if (!Profiling)
                {
                    modelValid.CopyWeightsFrom(model);
                    var validPerplexity = modelValid.RunEpoch(ptb.ValidData);
                    Console.WriteLine($"Epoch: {i + 1} Valid Perplexity: {validPerplexity:F3}");
                }
            }

            if (!Profiling)
            {
                modelTest.CopyWeightsFrom(model);
                Console.WriteLine("Testing with test data, this is slow, since batch size is set to small...");
                var testPerplexity = modelTest.RunEpoch(ptb.TestData, verbose: true);
                Console.WriteLine($"Test Perplexity: {testPerplexity:F3}");
            }

            if (!Profiling && isConsole)
            {
                var inputs = new int[cfgInteractive.NumSteps, 1];
                modelInteractive.CopyWeightsFrom(model);
                // since the entropy and softmax are merged , so we have to allocate the target (label) tensor
                // this could be improved , by adding some null checking?
                modelInteractive.Optimizer.AssignTensor(modelInteractive.Targets, inputs.AsTensor());

                while (true)
                {
                    Console.WriteLine();
                    Console.WriteLine($"Enter some words (less than {cfgInteractive.NumSteps} words)");
                    var readLine = Console.ReadLine();
                    if (readLine == null) break;
                    var line = readLine.Trim(' ', '\t', '\r', '\n');
                    var words = line.Split(new[] {' ', '\t', '\r', '\n'}, StringSplitOptions.RemoveEmptyEntries);
                    if (words.Length <= 0 || words.Length > cfgInteractive.NumSteps) continue;

                    for (var i = 0; i < cfgInteractive.NumSteps; ++i)
                    {
                        if (i < words.Length)
                        {
                            inputs[i, 0] = ptb.WordToId(words[i]);
                        }
                        else
                        {
                            inputs[i, 0] = ptb.WordToId("<unk>");
                        }
                    }

                    Console.WriteLine("Your inputs are:");
                    for (var i = 0; i < cfgInteractive.NumSteps; ++i)
                    {
                        Console.Write($"{ptb.IdToWord(inputs[i, 0])} ");
                    }
                    Console.WriteLine();

                    modelInteractive.ResetStates();
                    modelInteractive.Optimizer.AssignTensor(modelInteractive.Inputs, inputs.AsTensor());
                    modelInteractive.Optimizer.Forward();

                    var logPred = modelInteractive.Optimizer.GetTensor(modelInteractive.Loss.LogPred).ToArray2D();
                    var pred = new List<IndexAndProb>();
                    var totalProb = 0.0;
                    for (var i = 0; i < cfgInteractive.VocabSize; ++i)
                    {
                        var p = new IndexAndProb {Index = i, Prob = Math.Exp(logPred[words.Length - 1, i])};
                        pred.Add(p);
                        totalProb += p.Prob;
                    }
                    Console.WriteLine($"Total probability: {totalProb:F4}");
                    pred.Sort();
                    Console.WriteLine("Candidates are:");
                    pred.Take(10).Iter((x, o) => { Console.WriteLine($" {x.Prob:P2} --> {ptb.IdToWord(x.Index)}"); });
                }
            }
        }

        // This model uses cuDNN version
        public class Model2
        {
            public Model2(Context ctx, Config cfg, bool isTraining = true)
            {
                Config = cfg;
                IsTraining = isTraining;

                Inputs = Variable<int>(PartialShape.Create(Config.NumSteps, Config.BatchSize));
                Targets = Variable<int>(PartialShape.Create(Config.NumSteps, Config.BatchSize));

                Embedding = new Embedding<float>(Inputs, cfg.VocabSize, cfg.HiddenSize, initScale: cfg.InitScale);
                EmbeddedOutput = Embedding.Output;
                if (isTraining && cfg.KeepProb < 1.0)
                {
                    var dropout = new Dropout<float>(EmbeddedOutput, dropoutProb: 1.0 - cfg.KeepProb);
                    EmbeddedOutput = dropout.Output;
                }

                // rnn layer
                RNN = new RNN<float>(EmbeddedOutput, cfg.NumLayers, cfg.HiddenSize, isTraining: isTraining, dropout: isTraining && cfg.KeepProb < 1.0 ? 1.0 - Config.KeepProb : 0.0, bias: 0.0);
                RNNOutput = RNN.Y;
                if (isTraining && cfg.KeepProb < 1.0)
                {
                    var dropout = new Dropout<float>(RNNOutput, dropoutProb: 1.0 - cfg.KeepProb);
                    RNNOutput = dropout.Output;
                }

                FC = new FullyConnected<float>(RNNOutput.Reshape(RNNOutput.Shape[0]*RNNOutput.Shape[1], RNNOutput.Shape[2]), cfg.VocabSize);

                Loss = new SoftmaxCrossEntropySparse<float>(FC.Output, Targets.Reshape(Targets.Shape[0]*Targets.Shape[1]));

                Optimizer = new GradientDescentOptimizer(ctx, Loss.Loss, Config.LearningRate, new GlobalNormGradientClipper(Config.MaxGradNorm));

                // warmup (for JIT, and better timing measure)
                Optimizer.Initalize();
                ResetStates();
                Optimizer.AssignTensor(Inputs, Fill(Shape.Create(Inputs.Shape.AsArray), 0));
                Optimizer.AssignTensor(Targets, Fill(Shape.Create(Targets.Shape.AsArray), 0));
                Optimizer.Forward();
                if (isTraining)
                {
                    // TODO
                    Optimizer.Backward();
                }

                // now reset states
                Optimizer.Initalize();
                ResetStates();
            }

            public void CopyWeightsFrom(Model2 o)
            {
                Optimizer.AssignTensor(Embedding.Weights, o.Optimizer.GetTensor(o.Embedding.Weights));
                Optimizer.AssignTensor(RNN.W, o.Optimizer.GetTensor(o.RNN.W));
                Optimizer.AssignTensor(FC.Weights, o.Optimizer.GetTensor(o.FC.Weights));
                Optimizer.AssignTensor(FC.Bias, o.Optimizer.GetTensor(o.FC.Bias));
            }

            public void ResetStates()
            {
                Optimizer.AssignTensor(RNN.CX, Fill(Shape.Create(RNN.CX.Shape.AsArray), 0.0f));
                Optimizer.AssignTensor(RNN.HX, Fill(Shape.Create(RNN.HX.Shape.AsArray), 0.0f));
            }

            public void CopyStates()
            {
                Optimizer.AssignTensor(RNN.CX, Optimizer.GetTensor(RNN.CY));
                Optimizer.AssignTensor(RNN.HX, Optimizer.GetTensor(RNN.HY));
            }

            public double RunEpoch(int[] data, double learningRate = 1.0, bool verbose = false)
            {
                var epochSize = (data.Length/Config.BatchSize - 1)/Config.NumSteps;
                var time = Stopwatch.StartNew();
                var costs = 0.0;
                var iters = 0;
                var step = 0;
                var firstBatch = true;

                foreach (var batch in Data.Iterator(data, Config.NumSteps, Config.BatchSize))
                {
                    Optimizer.AssignTensor(Inputs, batch.Inputs.AsTensor());
                    Optimizer.AssignTensor(Targets, batch.Targets.AsTensor());

                    if (firstBatch)
                    {
                        // set h0 and c0 to 0 at each epoch start
                        ResetStates();
                        firstBatch = false;
                    }
                    else
                    {
                        CopyStates();
                    }

                    Optimizer.Forward();

                    if (IsTraining)
                    {
                        Optimizer.Backward();
                        Optimizer.Optimize(learningRate);
                    }

                    var loss = Optimizer.GetTensor(Loss.Loss).ToScalar();
                    var cost = loss/Config.BatchSize;
                    costs += cost;
                    iters += Config.NumSteps;

                    if (Profiling || (verbose && (step%(epochSize/10) == 10)))
                    {
                        var perplexity = Math.Exp(costs/iters);
                        var wps = (iters*Config.BatchSize)/(time.Elapsed.TotalMilliseconds/1000.0);

                        Console.WriteLine($"{step:D4}: {step*1.0/epochSize:F3} perplexity: {perplexity:F3} speed:{wps:F0} wps cost: {cost:F3}");
                    }

                    if (Profiling && step > 5) break;

                    step++;
                }
                return Math.Exp(costs/iters);
            }

            public Config Config { get; }

            public bool IsTraining { get; }

            public Variable<int> Inputs { get; }

            public Variable<int> Targets { get; }

            public Embedding<float> Embedding { get; }

            public Variable<float> EmbeddedOutput { get; }

            public RNN<float> RNN { get; }

            public Variable<float> RNNOutput { get; }

            public FullyConnected<float> FC { get; }

            public SoftmaxCrossEntropySparse<float> Loss { get; }

            public GradientDescentOptimizer Optimizer { get; }
        }

        [Test, Ignore("To long to run, please explicitly run it.")]
        public static void Run2()
        {
            Run2(false, CfgType);
        }

        public static void Run2(bool isConsole, ConfigType cfgType)
        {
            Console.WriteLine($"cuDNN Version, Config: {cfgType}");

            var ptb = new Data(DataPath);
            var ctx = Context.GpuContext(0);

            Config cfg, cfgValid, cfgTest, cfgInteractive;

            switch (cfgType)
            {
                case ConfigType.Small:
                    cfg = Config.Small(batchSize: 20);
                    cfgValid = Config.Small(batchSize: 20);
                    cfgTest = Config.Small(batchSize: 1, numSteps: 1);
                    cfgInteractive = Config.Small(batchSize: 1, numSteps: 10);
                    break;
                case ConfigType.Medium:
                    cfg = Config.Medium(batchSize: 20);
                    cfgValid = Config.Medium(batchSize: 20);
                    cfgTest = Config.Medium(batchSize: 1, numSteps: 1);
                    cfgInteractive = Config.Medium(batchSize: 1, numSteps: 10);
                    break;
                case ConfigType.Large:
                    cfg = Config.Large(batchSize: 20);
                    cfgValid = Config.Large(batchSize: 20);
                    cfgTest = Config.Large(batchSize: 1, numSteps: 1);
                    cfgInteractive = Config.Large(batchSize: 1, numSteps: 10);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(cfgType), cfgType, null);
            }

            Assert.AreEqual(ptb.WordToIdDict.Count, cfg.VocabSize);
            Assert.AreEqual(ptb.WordToIdDict.Count, cfgValid.VocabSize);
            Assert.AreEqual(ptb.WordToIdDict.Count, cfgTest.VocabSize);
            Assert.AreEqual(ptb.WordToIdDict.Count, cfgInteractive.VocabSize);

            var model = new Model2(ctx, cfg, isTraining: true);
            var modelValid = new Model2(ctx, cfgValid, isTraining: false);
            var modelTest = new Model2(ctx, cfgTest, isTraining: false);
            var modelInteractive = new Model2(ctx, cfgInteractive, isTraining: false);

            for (var i = 0; i < cfg.MaxMaxEpoch; ++i)
            {
                var lrDecay = Math.Pow(cfg.LrDecay, Math.Max(i - cfg.MaxEpoch, 0.0));
                var learningRate = cfg.LearningRate*lrDecay;

                Console.WriteLine($"Epoch: {i + 1} Learning rate: {learningRate:F3}");
                var trainPerplexity = model.RunEpoch(ptb.TrainData, learningRate: learningRate, verbose: true);
                Console.WriteLine($"Epoch: {i + 1} Train Perplexity: {trainPerplexity:F3}");

                if (!Profiling)
                {
                    modelValid.CopyWeightsFrom(model);
                    var validPerplexity = modelValid.RunEpoch(ptb.ValidData);
                    Console.WriteLine($"Epoch: {i + 1} Valid Perplexity: {validPerplexity:F3}");
                }
            }

            if (!Profiling)
            {
                modelTest.CopyWeightsFrom(model);
                Console.WriteLine("Testing with test data, this is slow, since batch size is set to small...");
                var testPerplexity = modelTest.RunEpoch(ptb.TestData, verbose: true);
                Console.WriteLine($"Test Perplexity: {testPerplexity:F3}");
            }

            if (!Profiling && isConsole)
            {
                var inputs = new int[cfgInteractive.NumSteps, 1];
                modelInteractive.CopyWeightsFrom(model);
                // since the entropy and softmax are merged , so we have to allocate the target (label) tensor
                // this could be improved , by adding some null checking?
                modelInteractive.Optimizer.AssignTensor(modelInteractive.Targets, inputs.AsTensor());

                while (true)
                {
                    Console.WriteLine();
                    Console.WriteLine($"Enter some words (less than {cfgInteractive.NumSteps} words)");
                    var readLine = Console.ReadLine();
                    if (readLine == null) break;
                    var line = readLine.Trim(' ', '\t', '\r', '\n');
                    var words = line.Split(new[] {' ', '\t', '\r', '\n'}, StringSplitOptions.RemoveEmptyEntries);
                    if (words.Length <= 0 || words.Length > cfgInteractive.NumSteps) continue;

                    for (var i = 0; i < cfgInteractive.NumSteps; ++i)
                    {
                        if (i < words.Length)
                        {
                            inputs[i, 0] = ptb.WordToId(words[i]);
                        }
                        else
                        {
                            inputs[i, 0] = ptb.WordToId("<unk>");
                        }
                    }

                    Console.WriteLine("Your inputs are:");
                    for (var i = 0; i < cfgInteractive.NumSteps; ++i)
                    {
                        Console.Write($"{ptb.IdToWord(inputs[i, 0])} ");
                    }
                    Console.WriteLine();

                    modelInteractive.ResetStates();
                    modelInteractive.Optimizer.AssignTensor(modelInteractive.Inputs, inputs.AsTensor());
                    modelInteractive.Optimizer.Forward();

                    var logPred = modelInteractive.Optimizer.GetTensor(modelInteractive.Loss.LogPred).ToArray2D();
                    var pred = new List<IndexAndProb>();
                    var totalProb = 0.0;
                    for (var i = 0; i < cfgInteractive.VocabSize; ++i)
                    {
                        var p = new IndexAndProb {Index = i, Prob = Math.Exp(logPred[words.Length - 1, i])};
                        pred.Add(p);
                        totalProb += p.Prob;
                    }
                    Console.WriteLine($"Total probability: {totalProb:F4}");
                    pred.Sort();
                    Console.WriteLine("Candidates are:");
                    pred.Take(10).Iter((x, o) => { Console.WriteLine($" {x.Prob:P2} --> {ptb.IdToWord(x.Index)}"); });
                }
            }
        }

        private static void Main()
        {
            if (Profiling)
            {
                Run1(false, CfgType);
                Run2(false, CfgType);
            }
            else
            {
                //Run1(true, CfgType);
                Run2(true, CfgType);
            }

            // This line is an alea bug, will be fixed with new alea release.
            Context.GpuContext(0).Dispose();
        }
    }
}
