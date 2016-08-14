using System;
using Alea.cuDNN;
using Alea.Parallel.Device;
using static AleaTK.Library;
using static AleaTK.ML.Library;

namespace AleaTK.ML.Operator
{
    public class SoftmaxCrossEntropy<T> : Differentiable
    {
        public SoftmaxCrossEntropy(Variable<T> input, Variable<T> label)
        {
            Input = input;
            Label = label;
            Pred = Variable<T>();
            Loss = Variable<T>();
            M = AuxVariable<T>();
            N = AuxVariable<T>();
            AddInput(input);
            AddInput(label);
            AddOutput(Pred);
            AddOutput(Loss);
            AddAuxVar(M);
            AddAuxVar(N);
        }

        public Variable<T> Input { get; }

        public Variable<T> Label { get; }

        public Variable<T> Pred { get; }

        public Variable<T> Loss { get; }

        public Variable<T> M { get; }

        public Variable<T> N { get; }

        public override void Forward(Executor executor)
        {
            var z = executor.GetTensor(Input);
            var y = executor.GetTensor(Label);

            // ---- old solution
            // pred is the output of softmax
            //executor.AssignTensor(LogPred, Exp(z) / ReduceSum(Exp(z).Reshape(-1, z.Shape[z.Shape.Rank - 1]), true, 1));

            // loss is the cross entropy
            //var p = executor.GetTensor(LogPred);
            //executor.AssignTensor(Loss, -ReduceMean(ReduceSum(y * Log(p).Reshape(-1, z.Shape[z.Shape.Rank - 1]), 1)));

            // ---- more stable solution
            executor.AssignTensor(M, ReduceMax(z.Reshape(-1, z.Shape[z.Shape.Rank - 1]), true, 1));
            var m = executor.GetTensor(M);

            executor.AssignTensor(N, z - m - Log(ReduceSum(Exp(z - m), true, 1)));
            var n = executor.GetTensor(N);

            executor.AssignTensor(Loss, -ReduceMean(ReduceSum(y*n, 1)));

            executor.AssignTensor(Pred, Exp(n));
        }

        public override void Backward(Executor executor)
        {
            var p = executor.GetTensor(Pred);
            var y = executor.GetTensor(Label);
            executor.AssignGradient(Input, p - y);
        }
    }

    public class SoftmaxCrossEntropySparse<T> : Differentiable
    {
        public SoftmaxCrossEntropySparse(Variable<T> input, Variable<int> label)
        {
            Input = input;
            Label = label;
            LogPred = Variable<T>();
            Loss = Variable<T>();
            Temp = AuxVariable<T>();
            M = AuxVariable<T>();
            N = AuxVariable<T>();
            AddInput(input);
            AddInput(label);
            AddOutput(LogPred);
            AddOutput(Loss);
            AddAuxVar(Temp);
            AddAuxVar(M);
            AddAuxVar(N);
        }

        public Variable<T> Input { get; }

        public Variable<int> Label { get; }

        public Variable<T> LogPred { get; }

        public Variable<T> Loss { get; }

        public Variable<T> Temp { get; }

        public Variable<T> M { get; }

        public Variable<T> N { get; }

        public override void Forward(Executor executor)
        {
            var z = executor.GetTensor(Input);
            var y = executor.GetTensor(Label);

            Util.EnsureTrue(z.Shape.Rank == 2);
            Util.EnsureTrue(Dnn.IsAvailable, "TODO: make non-cuDnn implementation.");

            var n = (int)z.Shape[0];
            var classes = (int)z.Shape[1];

            using (var xDesc = executor.TensorDescRepo.Acquire())
            using (var yDesc = executor.TensorDescRepo.Acquire())
            {
                var dnn = executor.Context.ToGpuContext().Dnn;
                xDesc.Value.SetND(Dnn.DataTypeOf(typeof(T)), new[] { n, classes, 1, 1 }, new[] { classes, 1, 1, 1 });
                yDesc.Value.SetND(Dnn.DataTypeOf(typeof(T)), new[] { n, classes, 1, 1 }, new[] { classes, 1, 1, 1 });

                var xPtr = executor.GetTensor(Input).Buffer.Ptr;
                var yPtr = executor.GetTensor(LogPred, Shape.Create(n, classes)).Buffer.Ptr;
                var alpha = ScalarOps.Conv<T>(1.0);
                var beta = ScalarOps.Conv<T>(0.0);
                const SoftmaxAlgorithm algorithm = SoftmaxAlgorithm.LOG;
                const SoftmaxMode mode = SoftmaxMode.INSTANCE;

                dnn.SoftmaxForward(algorithm, mode, alpha, xDesc.Value, xPtr, beta, yDesc.Value, yPtr);
            }

            // TODO: make it expression
            var logPred = executor.GetTensor(LogPred);
            var temp = executor.GetTensor(Temp, Shape.Create(n));

            var ctx = executor.Context;

            if (ctx.Type == ContextType.Gpu && logPred.Layout.IsInnerChangeMostFullyPacked)
            {
                var stream = ctx.ToGpuContext().Stream;
                var tempPtr = temp.Buffer.Ptr;
                var logPredPtr = logPred.Buffer.Ptr;
                var idxPtr = y.Buffer.Ptr;
                DeviceFor.For(stream, 0, n, i =>
                {
                    var idx = idxPtr[i];
                    tempPtr[i] = logPredPtr[i * classes + idx];
                });
                executor.AssignTensor(Loss, -ReduceSum(temp));
                return;
            }

            throw new NotImplementedException();
        }

        public override void Backward(Executor executor)
        {
            var p = executor.GetTensor(LogPred);
            var y = executor.GetTensor(Label);

            Util.EnsureTrue(p.Shape.Rank == 2);
            var n = (int) p.Shape[0];
            var classes = (int) p.Shape[1];

            executor.AssignGradient(Input, Exp(p));

            var g = executor.GetGradient(Input);

            var ctx = executor.Context;

            if (ctx.Type == ContextType.Gpu)
            {
                var stream = ctx.ToGpuContext().Stream;

                if (typeof (T) == typeof (float))
                {
                    var gptr = g.Buffer.Ptr.Reinterpret<float>();
                    var idxptr = y.Buffer.Ptr;
                    DeviceFor.For(stream, 0, n, i =>
                    {
                        var idx = idxptr[i];
                        gptr[i*classes + idx] -= 1.0f;
                    });

                    return;
                }
                else if (typeof(T) == typeof(double))
                {
                    var gptr = g.Buffer.Ptr.Reinterpret<double>();
                    var idxptr = y.Buffer.Ptr;
                    DeviceFor.For(stream, 0, n, i =>
                    {
                        var idx = idxptr[i];
                        gptr[i * classes + idx] -= 1.0;
                    });

                    return;
                }
                else
                {
                    throw new NotImplementedException();
                }
            }

            throw new NotImplementedException();
        }
    }

    public class Softmax<T> : Differentiable
    {
        public Softmax(Variable<T> input)
        {
            Input = input;
            Output = Variable<T>();
            AddInput(Input);
            AddOutput(Output);
        }

        public Variable<T> Input { get; }

        public Variable<T> Output { get; }

        public override void Forward(Executor executor)
        {
            var ctx = executor.Context;
            var x = executor.GetTensor(Input);
            var y = executor.GetTensor(Output, x.Shape);
            
            if (ctx.Type == ContextType.Gpu && x.Layout.IsInnerChangeMostFullyPacked)
            {
                var dnn = ctx.ToGpuContext().Dnn;
                var n = (int)x.Shape[0];
                var classes = (int)x.Shape[1];

                using (var xDesc = executor.TensorDescRepo.Acquire())
                using (var yDesc = executor.TensorDescRepo.Acquire())
                {
                    xDesc.Value.SetND(Dnn.DataTypeOf(typeof(T)), new[] { n, classes, 1, 1 }, new[] { classes, 1, 1, 1 });
                    yDesc.Value.SetND(Dnn.DataTypeOf(typeof(T)), new[] { n, classes, 1, 1 }, new[] { classes, 1, 1, 1 });

                    var xPtr = x.Buffer.Ptr;
                    var yPtr = y.Buffer.Ptr;
                    var alpha = ScalarOps.Conv<T>(1.0);
                    var beta = ScalarOps.Conv<T>(0.0);
                    const SoftmaxAlgorithm algorithm = SoftmaxAlgorithm.ACCURATE;
                    const SoftmaxMode mode = SoftmaxMode.INSTANCE;

                    dnn.SoftmaxForward(algorithm, mode, alpha, xDesc.Value, xPtr, beta, yDesc.Value, yPtr);
                }

                return;
            }

            throw new NotImplementedException();
        }

        public override void Backward(Executor executor)
        {
            var ctx = executor.Context;
            var x = executor.GetTensor(Input);
            var y = executor.GetTensor(Output);
            var dx = executor.GetGradient(Input, x.Shape);
            var dy = executor.GetGradient(Output);

            if (ctx.Type == ContextType.Gpu && x.Layout.IsInnerChangeMostFullyPacked)
            {
                var dnn = ctx.ToGpuContext().Dnn;
                var n = (int)x.Shape[0];
                var classes = (int)x.Shape[1];

                using (var xDesc = executor.TensorDescRepo.Acquire())
                using (var yDesc = executor.TensorDescRepo.Acquire())
                {
                    xDesc.Value.SetND(Dnn.DataTypeOf(typeof(T)), new[] { n, classes, 1, 1 }, new[] { classes, 1, 1, 1 });
                    yDesc.Value.SetND(Dnn.DataTypeOf(typeof(T)), new[] { n, classes, 1, 1 }, new[] { classes, 1, 1, 1 });

                    var dxPtr = dx.Buffer.Ptr;
                    var yPtr = y.Buffer.Ptr;
                    var dyPtr = dy.Buffer.Ptr;
                    var alpha = ScalarOps.Conv<T>(1.0);
                    var beta = ScalarOps.Conv<T>(0.0);
                    const SoftmaxAlgorithm algorithm = SoftmaxAlgorithm.ACCURATE;
                    const SoftmaxMode mode = SoftmaxMode.INSTANCE;

                    dnn.SoftmaxBackward(algorithm, mode, alpha, yDesc.Value, yPtr, yDesc.Value, dyPtr, beta, xDesc.Value, dxPtr);
                }

                return;
            }

            throw new NotImplementedException();
        }
    }
}
