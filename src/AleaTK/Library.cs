using System;
using System.Linq;
using System.Security.Policy;
using System.Threading.Tasks;
using Alea;
using AleaTK.ExprImpl;

namespace AleaTK
{
    public static class Library
    {
        public static readonly ExprRegistry ExprRegistry = new ExprRegistry();

        static Library()
        {
            #region Neg
            ExprRegistry.Register<double>(
                (exprParam, inputExprs) => Map(inputExprs[0].CastExpr<double>(), ScalarOps.Neg, OpCodes.Neg),
                OpCodes.Neg, typeof(double));

            ExprRegistry.Register<float>(
                (exprParam, inputExprs) => Map(inputExprs[0].CastExpr<float>(), ScalarOps.Neg, OpCodes.Neg),
                OpCodes.Neg, typeof(float));
            #endregion

            #region Add
            ExprRegistry.Register<double>(
                (exprParam, inputExprs) => Map(inputExprs[0].CastExpr<double>(), inputExprs[1].CastExpr<double>(), ScalarOps.Add, OpCodes.Add),
                OpCodes.Add, typeof(double), typeof(double));

            ExprRegistry.Register<float>(
                (exprParam, inputExprs) => Map(inputExprs[0].CastExpr<float>(), inputExprs[1].CastExpr<float>(), ScalarOps.Add, OpCodes.Add),
                OpCodes.Add, typeof(float), typeof(float));
            #endregion

            #region Sub
            ExprRegistry.Register<double>(
                (exprParam, inputExprs) => Map(inputExprs[0].CastExpr<double>(), inputExprs[1].CastExpr<double>(), ScalarOps.Sub, OpCodes.Sub),
                OpCodes.Sub, typeof(double), typeof(double));

            ExprRegistry.Register<float>(
                (exprParam, inputExprs) => Map(inputExprs[0].CastExpr<float>(), inputExprs[1].CastExpr<float>(), ScalarOps.Sub, OpCodes.Sub),
                OpCodes.Sub, typeof(float), typeof(float));
            #endregion

            #region Mul
            ExprRegistry.Register<double>(
                (exprParam, inputExprs) => Map(inputExprs[0].CastExpr<double>(), inputExprs[1].CastExpr<double>(), ScalarOps.Mul, OpCodes.Mul),
                OpCodes.Mul, typeof(double), typeof(double));

            ExprRegistry.Register<float>(
                (exprParam, inputExprs) => Map(inputExprs[0].CastExpr<float>(), inputExprs[1].CastExpr<float>(), ScalarOps.Mul, OpCodes.Mul),
                OpCodes.Mul, typeof(float), typeof(float));
            #endregion

            #region Div
            ExprRegistry.Register<double>(
                (exprParam, inputExprs) => Map(inputExprs[0].CastExpr<double>(), inputExprs[1].CastExpr<double>(), ScalarOps.Div, OpCodes.Div),
                OpCodes.Div, typeof(double), typeof(double));

            ExprRegistry.Register<float>(
                (exprParam, inputExprs) => Map(inputExprs[0].CastExpr<float>(), inputExprs[1].CastExpr<float>(), ScalarOps.Div, OpCodes.Div),
                OpCodes.Div, typeof(float), typeof(float));
            #endregion

            #region Exp
            ExprRegistry.Register<double>(
                (exprParam, inputExprs) => Map(inputExprs[0].CastExpr<double>(), DeviceFunction.Exp, OpCodes.Exp),
                OpCodes.Exp, typeof(double));

            ExprRegistry.Register<float>(
                (exprParam, inputExprs) => Map(inputExprs[0].CastExpr<float>(), DeviceFunction.Exp, OpCodes.Exp),
                OpCodes.Exp, typeof(float));
            #endregion

            #region Log
            ExprRegistry.Register<double>(
                (exprParam, inputExprs) => Map(inputExprs[0].CastExpr<double>(), DeviceFunction.Log, OpCodes.Log),
                OpCodes.Log, typeof(double));

            ExprRegistry.Register<float>(
                (exprParam, inputExprs) => Map(inputExprs[0].CastExpr<float>(), DeviceFunction.Log, OpCodes.Log),
                OpCodes.Log, typeof(float));
            #endregion

            ExprRegistry.Register<double>(
                (exprParam, inputExprs) => Map(inputExprs[0].CastExpr<double>(), DeviceFunction.Tanh, OpCodes.Tanh),
                OpCodes.Tanh, typeof(double));

            ExprRegistry.Register<float>(
                (exprParam, inputExprs) => Map(inputExprs[0].CastExpr<float>(), DeviceFunction.Tanh, OpCodes.Tanh),
                OpCodes.Tanh, typeof(float));

            #region Dot
            ExprRegistry.Register<double>(
                (exprParam, inputExprs) => new DotExpr<double>(inputExprs[0].CastExpr<double>(), inputExprs[1].CastExpr<double>(), 0.0, ScalarOps.Add, ScalarOps.Mul),
                OpCodes.Dot, typeof(double), typeof(double));

            ExprRegistry.Register<float>(
                (exprParam, inputExprs) => new DotExpr<float>(inputExprs[0].CastExpr<float>(), inputExprs[1].CastExpr<float>(), 0.0f, ScalarOps.Add, ScalarOps.Mul),
                OpCodes.Dot, typeof(float), typeof(float));
            #endregion

            #region Max
            ExprRegistry.Register<double>(
                (exprParam, inputExprs) =>
                    Map(inputExprs[0].CastExpr<double>(), inputExprs[1].CastExpr<double>(), DeviceFunction.Max,
                        OpCodes.Max),
                OpCodes.Max, typeof(double), typeof(double));

            ExprRegistry.Register<float>(
                (exprParam, inputExprs) =>
                    Map(inputExprs[0].CastExpr<float>(), inputExprs[1].CastExpr<float>(), DeviceFunction.Max,
                        OpCodes.Max),
                OpCodes.Max, typeof(float), typeof(float));
            #endregion

            #region Reduce
            ExprRegistry.Register<double>(
                (exprParam, inputExprs) =>
                {
                    bool keepDims = exprParam.KeepDims;
                    int[] reductionIndices = exprParam.ReductionIndices;
                    return new ReduceExpr<double>(inputExprs[0].CastExpr<double>(), ScalarOps.Add, keepDims,
                        reductionIndices, OpCodes.ReduceSum);
                },
                OpCodes.ReduceSum, typeof(double));

            ExprRegistry.Register<float>(
                (exprParam, inputExprs) =>
                {
                    bool keepDims = exprParam.KeepDims;
                    int[] reductionIndices = exprParam.ReductionIndices;
                    return new ReduceExpr<float>(inputExprs[0].CastExpr<float>(), ScalarOps.Add, keepDims,
                        reductionIndices, OpCodes.ReduceSum);
                },
                OpCodes.ReduceSum, typeof(float));

            ExprRegistry.Register<float>(
                (exprParam, inputExprs) =>
                {
                    bool keepDims = exprParam.KeepDims;
                    int[] reductionIndices = exprParam.ReductionIndices;
                    return new ReduceExpr<float>(inputExprs[0].CastExpr<float>(), DeviceFunction.Max, keepDims,
                        reductionIndices, OpCodes.ReduceMax);
                },
                OpCodes.ReduceMax, typeof(float));
            #endregion

            ExprRegistry.Register<float>(
                (exprParam, inputExprs) =>
                {
                    uint threshold = exprParam.Threshold;
                    double scale = exprParam.Scale;
                    return Map(inputExprs[0].CastExpr<float>(), inputExprs[1].CastExpr<uint>(),
                        (invalue, mask) => (float)(mask > threshold ? invalue * scale : 0.0f), OpCodes.Dropout);
                }, OpCodes.Dropout, typeof(float), typeof(uint));

            ExprRegistry.Register<double>(
                (exprParam, inputExprs) =>
                {
                    uint threshold = exprParam.Threshold;
                    double scale = exprParam.Scale;
                    return Map(inputExprs[0].CastExpr<double>(), inputExprs[1].CastExpr<uint>(),
                        (invalue, mask) => (double)(mask > threshold ? invalue * scale : 0.0), OpCodes.Dropout);
                }, OpCodes.Dropout, typeof(double), typeof(uint));

            ExprRegistry.Register<double>(
                (exprParam, inputExprs) =>
                    Map(inputExprs[0].CastExpr<double>(), x => x > 0.0 ? 1.0 : 0.0, OpCodes.ReLUGrad),
                OpCodes.ReLUGrad, typeof(double));

            ExprRegistry.Register<float>(
                (exprParam, inputExprs) =>
                    Map(inputExprs[0].CastExpr<float>(), x => x > 0.0f ? 1.0f : 0.0f, OpCodes.ReLUGrad),
                OpCodes.ReLUGrad, typeof(float));

            ExprRegistry.Register<double>(
                (exprParam, inputExprs) =>
                    Map(inputExprs[0].CastExpr<double>(), DeviceFunction.Sqrt, OpCodes.Sqrt),
                OpCodes.Sqrt, typeof(double));

            ExprRegistry.Register<float>(
                (exprParam, inputExprs) =>
                    Map(inputExprs[0].CastExpr<float>(), DeviceFunction.Sqrt, OpCodes.Sqrt),
                OpCodes.Sqrt, typeof(float));

            ExprRegistry.Register<float>(
                (exprParam, inputExprs) =>
                {
                    Func<float, float, float> add = ScalarOps.Add;
                    return new TakeGradExpr<float>(inputExprs[0].CastExpr<int>(), inputExprs[1].CastExpr<float>(),
                        exprParam.SourceRows, 0.0f, add);
                },
                OpCodes.TakeGrad, typeof(int), typeof(float));
        }

        public static Range Range(int beginInclusive, int endExclusive)
        {
            return AleaTK.Range.Create(beginInclusive, endExclusive);
        }

        #region Tensor creation (allocate or reference)
        public static Tensor<T> Allocate<T>(this Device device, Layout layout, long length)
        {
            var buffer = Buffer.Allocate<T>(device, layout, length);
            return new Tensor<T>(buffer);
        }

        public static Tensor<T> Allocate<T>(this Device device, Shape shape)
        {
            var layout = new Layout(shape);
            return device.Allocate<T>(layout, shape.Length);
        }

        public static Tensor<T> Reference<T>(Array array, Shape shape)
        {
            Util.EnsureEqual(shape.Length, array.LongLength);
            var buffer = Buffer.Reference<T>(array, shape);
            return new Tensor<T>(buffer);
        }

        public static Tensor<T> Reference<T>(Array array)
        {
            var shape = Shape.GetArrayShape(array);
            return Reference<T>(array, shape);
        }

        public static Tensor<T> AsTensor<T>(this T[] array)
        {
            return Reference<T>(array);
        }

        public static Tensor<T> AsTensor<T>(this T[] array, Shape shape)
        {
            return Reference<T>(array, shape);
        }

        public static Tensor<T> AsTensor<T>(this T[,] array)
        {
            return Reference<T>(array);
        }

        public static Tensor<T> AsTensor<T>(this T[,] array, Shape shape)
        {
            return Reference<T>(array, shape);
        }

        public static Tensor<T> AsTensor<T>(this T[,,] array)
        {
            return Reference<T>(array);
        }

        public static Tensor<T> AsTensor<T>(this T[,,] array, Shape shape)
        {
            return Reference<T>(array, shape);
        }
        #endregion

        #region Tensor initalization
        public static Tensor<T> Allocate<T>(this Context context, Shape shape, T initValue)
        {
            var tensor = context.Device.Allocate<T>(shape);
            context.Assign(tensor, initValue.AsScalar());
            return tensor;
        }

        public static Tensor<T> Allocate<T>(this Context context, T[] initValues)
        {
            var shape = Shape.GetArrayShape(initValues);
            var tensor = context.Device.Allocate<T>(shape);
            var initTensor = initValues.AsTensor();
            context.Copy(tensor, initTensor);
            return tensor;
        }

        public static Tensor<T> Allocate<T>(this Context context, T[,] initValues)
        {
            var shape = Shape.GetArrayShape(initValues);
            var tensor = context.Device.Allocate<T>(shape);
            var initTensor = initValues.AsTensor();
            context.Copy(tensor, initTensor);
            return tensor;
        }

        public static Tensor<T> Allocate<T>(this Context context, T[,,] initValues)
        {
            var shape = Shape.GetArrayShape(initValues);
            var tensor = context.Device.Allocate<T>(shape);
            var initTensor = initValues.AsTensor();
            context.Copy(tensor, initTensor);
            return tensor;
        }
        #endregion

        #region Tensor updater (assign and copy)

        public static Task Assign(this Context context, IValue tensor, Expr expr)
        {
            return Assignment.Run(context, tensor, expr);
        }

        public static Task Assign<T>(this Context context, Tensor<T> tensor, Expr<T> expr)
        {
            return Assignment.Run(context, tensor, expr);
        }

        public static Task Assign<T>(this Context context, Tensor<T> tensor, T[] newValues)
        {
            var newValuesTensor = newValues.AsTensor();
            return context.Copy(tensor, newValuesTensor);
        }

        public static Task Assign<T>(this Context context, Tensor<T> tensor, T scalar)
        {
            return context.Assign(tensor, scalar.AsScalar());
        }

        public static Task Copy<T>(this Context context, Tensor<T> dstTensor, Tensor<T> srcTensor)
        {
            if (!srcTensor.Layout.Shape.SequenceEqual(dstTensor.Layout.Shape))
            {
                throw new InvalidOperationException($"Copy require same shape! dst({dstTensor.Shape}) src({srcTensor.Shape})");
            }

            // if both are on cpu side, then we cannot copy, we have to assign it.
            if (dstTensor.Device == Device.CpuDevice && srcTensor.Device == Device.CpuDevice)
            {
                return Context.CpuContext.Assign(dstTensor, srcTensor);
            }

            // if strides are same, then we can safely copy it, the buffer length should be correct,
            // we cannot tell if buffer length is correct here.
            if (srcTensor.Layout.Strides.SequenceEqual(dstTensor.Layout.Strides))
            {
                var dstMemory = dstTensor.Memory.Memory;
                var srcMemory = srcTensor.Memory.Memory;
                var dstOffset = dstTensor.Memory.Offset;
                var srcOffset = srcTensor.Memory.Offset;
                var dstLength = dstTensor.Memory.Length;
                var srcLength = srcTensor.Memory.Length;
                var length = Math.Min(dstLength, srcLength);
                Task task;

                if (context.Type == ContextType.Gpu && (context.Device == dstTensor.Device || context.Device == srcTensor.Device))
                {
                    var stream = context.ToGpuContext().Stream;
                    task = new Task(() => Memory.XiangCopy2(stream, srcMemory, srcOffset, dstMemory, dstOffset, length));
                }
                else
                {
                    task = new Task(() => Memory.XiangCopy(srcMemory, srcOffset, dstMemory, dstOffset, length));
                }

                // TODO:@RDE
                task.Start();
                task.Wait();
                return task;
            }

            throw new NotImplementedException();
        }
        #endregion

        #region Tensor data retriver
        public static void Print<T>(this Context context, Tensor<T> tensor, bool all = false)
        {
            Task task;

            // print must work on cpu side, if the tensor is on cpu side already, we
            // can directly print it (using RDE to sync the resource)
            if (tensor.Device == Device.CpuDevice)
            {
                task = new Task(() => tensor.Layout.Print(tensor.Buffer.RawReader, all));
            }
            else
            {
                // if the tensor is not on cpu, then we need make a copy of it and print it.
                var cpuTensor = Device.CpuDevice.Allocate<T>(tensor.Layout, tensor.Memory.Length);
                context.Copy(cpuTensor, tensor).Wait();
                task = new Task(() => tensor.Layout.Print(cpuTensor.Buffer.RawReader, all));
            }

            // TODO:@RDE
            task.Start();
            task.Wait();
        }

        public static void Print<T>(this Tensor<T> tensor, bool all = false)
        {
            Context.CpuContext.Print(tensor, all);
        }

        public static T[] ToArray<T>(this Context context, Tensor<T> tensor)
        {
            // if the tensor's layout is C Style, then we just make one copy and return it
            if (tensor.Layout.IsInnerChangeMostFullyPacked)
            {
                var layout = tensor.Layout;
                var shape = layout.Shape;
                var array = new T[shape.Length];
                var cpuTensor = array.AsTensor(shape);
                context.Copy(cpuTensor, tensor).Wait();
                return array;
            }

            // not c style layout, we need allocate a temp tensor on that device, assign it, then copy it back
            throw new NotImplementedException();
        }

        public static T[] ToArray<T>(this Tensor<T> tensor)
        {
            return Context.CpuContext.ToArray(tensor);
        }

        public static T[,,] ToArray3D<T>(this Context context, Tensor<T> tensor)
        {
            Util.EnsureTrue(tensor.Layout.Rank >= 3);

            if (!tensor.Layout.IsInnerChangeMostFullyPacked)
            {
                var tempTensor = context.Device.Allocate<T>(tensor.Layout, tensor.Buffer.Memory.Length);
                context.Copy(tempTensor, tensor).Wait();
                tensor = tempTensor;
            }

            var l0 = tensor.Layout.Shape[0];
            var l1 = tensor.Layout.Shape[1];
            var l2 = tensor.Layout.Shape.Skip(2).Aggregate(ScalarOps.Mul);
            var array = new T[l0, l1, l2];
            var cpuTensor = array.AsTensor();
            context.Copy(cpuTensor, tensor).Wait();
            return array;
        }

        public static T[,,] ToArray3D<T>(this Tensor<T> tensor)
        {
            return Context.CpuContext.ToArray3D(tensor);
        }

        public static T[,] ToArray2D<T>(this Context context, Tensor<T> tensor)
        {
            Util.EnsureTrue(tensor.Layout.Rank >= 2);

            if (!tensor.Layout.IsInnerChangeMostFullyPacked)
            {
                var tempTensor = context.Device.Allocate<T>(tensor.Layout, tensor.Buffer.Memory.Length);
                context.Copy(tempTensor, tensor).Wait();
                tensor = tempTensor;
            }

            var rows = tensor.Layout.Shape[0];
            var cols = tensor.Layout.Shape.Skip(1).Aggregate(ScalarOps.Mul);
            var array = new T[rows, cols];
            var cpuTensor = array.AsTensor();
            context.Copy(cpuTensor, tensor).Wait();
            return array;
        }

        public static T[,] ToArray2D<T>(this Tensor<T> tensor)
        {
            return Context.CpuContext.ToArray2D(tensor);
        }

        public static T ToScalar<T>(this Context context, Tensor<T> tensor)
        {
            var array = context.ToArray(tensor);
            return array[0];
        }

        public static T ToScalar<T>(this Tensor<T> tensor)
        {
            return Context.CpuContext.ToScalar(tensor);
        }

        public static Tensor<T> Eval<T>(this Context context, Expr<T> expr)
        {
            var tensor = context.Device.Allocate<T>(expr.Shape);
            context.Assign(tensor, expr);
            return tensor;
        }
        #endregion

        #region Expressions
        public static Expr<T> AsScalar<T>(this T value)
        {
            return new ScalarExpr<T>(value);
        }

        public static Expr<T> AsScalar<T>(this long value)
        {
            if (typeof (T) == typeof (double))
            {
                var convertedValue = (double) value;
                return new ScalarExpr<T>((T)((object)convertedValue));
            }

            if (typeof(T) == typeof(float))
            {
                var convertedValue = (float)value;
                return new ScalarExpr<T>((T)((object)convertedValue));
            }

            throw new InvalidOperationException($"Type not supported: long to {typeof(T)}");
        }

        public static Expr<T> AsScalar<T>(this double value)
        {
            return AsScalar(value, typeof (T)).CastExpr<T>();
        }

        public static Expr AsScalar(this double value, Type dataType)
        {
            if (dataType == typeof (double))
            {
                return new ScalarExpr<double>(value);
            }

            if (dataType == typeof (float))
            {
                return new ScalarExpr<float>((float)value);
            }

            throw new InvalidOperationException("Type not supported");
        }

        public static Expr<T> Reshape<T>(this Expr<T> input, params long[] dims)
        {
            return new ReShapeExpr<T>(input, dims);
        }

        public static Expr<T> Fill<T>(Shape shape, T value)
        {
            return new ScalarExpr<T>(shape, value);
        }

        public static Expr<TResult> Map<TInput, TResult>(Expr<TInput> input, Func<TInput, TResult> transform, string opCode = OpCodes.Map1)
        {
            return new Map1Expr<TInput, TResult>(input, transform, opCode);
        }

        public static Expr<TResult> Map<TInput1, TInput2, TResult>(Expr<TInput1> input1, Expr<TInput2> input2, Func<TInput1, TInput2, TResult> transform, string opCode = OpCodes.Map2)
        {
            return new Map2Expr<TInput1, TInput2, TResult>(input1, input2, transform, opCode);
        }

        public static Expr<T> Tanh<T>(Expr<T> a)
        {
            return ExprRegistry.Create<T>(OpCodes.Tanh, a);
        }

        public static Expr<T> Exp<T>(Expr<T> a)
        {
            return ExprRegistry.Create<T>(OpCodes.Exp, a);
        }

        public static Expr<T> Log<T>(Expr<T> a)
        {
            return ExprRegistry.Create<T>(OpCodes.Log, a);
        }

        public static Expr<T> Sqrt<T>(Expr<T> a)
        {
            return ExprRegistry.Create<T>(OpCodes.Sqrt, a);
        }

        public static Expr<T> Max<T>(Expr<T> a, Expr<T> b)
        {
            return ExprRegistry.Create<T>(OpCodes.Max, a, b);
        }

        public static Expr<T> ReLUGrad<T>(Expr<T> a)
        {
            return ExprRegistry.Create<T>(OpCodes.ReLUGrad, a);
        }

        public static Expr<T> ReduceSum<T>(Expr<T> a, bool keepDims, params int[] reductionIndices)
        {
            var p = new { KeepDims = keepDims, ReductionIndices = reductionIndices};
            return ExprRegistry.Create<T>(OpCodes.ReduceSum, p, a);
        }

        public static Expr<T> ReduceSum<T>(Expr<T> a, params int[] reductionIndices)
        {
            return ReduceSum(a, false, reductionIndices);
        }

        public static Expr<T> ReduceMax<T>(Expr<T> a, bool keepDims, params int[] reductionIndices)
        {
            var p = new { KeepDims = keepDims, ReductionIndices = reductionIndices };
            return ExprRegistry.Create<T>(OpCodes.ReduceMax, p, a);
        }

        public static Expr<T> ReduceMax<T>(Expr<T> a, params int[] reductionIndices)
        {
            return ReduceMax(a, false, reductionIndices);
        }

        public static Expr<T> ReduceMean<T>(Expr<T> a, bool keepDims, params int[] reductionIndices)
        {
            return ReduceSum(a, keepDims, reductionIndices) / (a.Shape.Length.AsScalar<T>());
        }

        public static Expr<T> ReduceMean<T>(Expr<T> a, params int[] reductionIndices)
        {
            return ReduceSum(a, reductionIndices) / (a.Shape.Length.AsScalar<T>());
        }

        public static Expr<T> RandomUniform<T>(Shape shape = null, ulong? seed = null, ulong offset = 0UL,
            PseudoRandomType type = PseudoRandomType.Default)
        {
            seed = seed ?? ((ulong) DateTime.Now.Ticks);
            return new PseudoRandomExpr<T>(shape, type, new UniformDistribution(), seed.Value, offset);
        }

        public static Expr<T> RandomUniform<T>(double from, double to, Shape shape = null, ulong? seed = null, ulong offset = 0UL,
            PseudoRandomType type = PseudoRandomType.Default)
        {
            seed = seed ?? ((ulong)DateTime.Now.Ticks);
            var width = (to - from).AsScalar<T>();
            var origin = from.AsScalar<T>();
            var prngExpr = new PseudoRandomExpr<T>(shape, type, new UniformDistribution(), seed.Value, offset);
            return origin + width * prngExpr;
        }

        public static Expr<T> RandomNormal<T>(Shape shape = null, ulong? seed = null, ulong offset = 0UL,
            double mean = 0.0, double stddev = 1.0, PseudoRandomType type = PseudoRandomType.Default)
        {
            seed = seed ?? ((ulong)DateTime.Now.Ticks);
            return new PseudoRandomExpr<T>(shape, type, new NormalDistribution(mean, stddev), seed.Value, offset);
        }

        public static Expr<T> Dot<T>(Expr<T> a, Expr<T> b)
        {
            return ExprRegistry.Create<T>(OpCodes.Dot, a, b);
        }

        public static Expr<T> Take<T>(Expr<int> indices, Expr<T> source)
        {
            return new TakeExpr<T>(indices, source);
        }

        public static Expr<T> TakeGrad<T>(Expr<int> indices, Expr<T> outputGrad, int sourceRows)
        {
            return ExprRegistry.Create<T>(OpCodes.TakeGrad, new {SourceRows = sourceRows}, indices, outputGrad);
        }

        public static Expr<T> Dropout<T>(Expr<T> invalue, Expr<uint> mask, uint threshold, double scale)
        {
            return ExprRegistry.Create<T>(OpCodes.Dropout, new {Threshold = threshold, Scale = scale}, invalue,
                mask);
        }
        #endregion
    }
}
