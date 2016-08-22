using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AleaTK;
using AleaTK.ML;
using AleaTK.ML.Operator;
using NUnit.Framework;
using static AleaTK.ML.Library;

namespace AleaTKTest
{
    public static class Experiment
    {
        // this is just a workaround here, it can be moved to framework later
        public static void AssignOrSetTensor<T>(Executor executor, Variable<T> var, Tensor<T> tensor)
        {
            if (tensor.Device == executor.Context.Device)
            {
                executor.SetTensor(var, tensor);
            }
            else
            {
                executor.AssignTensor(var, tensor);
            }
        }

        // this is just a workaround here, it can be moved to framework later
        public static void AssignOrSetGradient<T>(Executor executor, Variable<T> var, Tensor<T> gradient)
        {
            if (gradient.Device == executor.Context.Device)
            {
                executor.SetGradient(var, gradient);
            }
            else
            {
                executor.AssignGradientDirectly(var, gradient);
            }
        }

        // This is a compile function, given a function, which takes variable as input, and return one variable for output
        // We can create many overriding method here for different argument numbers
        public static Tuple<Func<Tensor<T1>, Tensor<T2>, Tensor<T3>, Tensor<TR>>, Func<Tensor<TR>, Tuple<Tensor<T1>, Tensor<T2>, Tensor<T3>>>>
            Compile<T1, T2, T3, TR>(Context ctx, Func<Variable<T1>, Variable<T2>, Variable<T3>, Variable<TR>> function)
        {
            var var1 = Variable<T1>();
            var var2 = Variable<T2>();
            var var3 = Variable<T3>();
            var varR = function(var1, var2, var3);
            var executor = new Executor(ctx, varR) { AssignAllGradient = true };

            Func<Tensor<T1>, Tensor<T2>, Tensor<T3>, Tensor<TR>> forward =
                (tensor1, tensor2, tensor3) =>
                {
                    AssignOrSetTensor(executor, var1, tensor1);
                    AssignOrSetTensor(executor, var2, tensor2);
                    AssignOrSetTensor(executor, var3, tensor3);
                    executor.Forward();
                    return executor.GetTensor(varR);
                };

            Func<Tensor<TR>, Tuple<Tensor<T1>, Tensor<T2>, Tensor<T3>>> backward =
                gradientR =>
                {
                    AssignOrSetGradient(executor, varR, gradientR);
                    executor.Backward();
                    var gradient1 = executor.GetGradient(var1);
                    var gradient2 = executor.GetGradient(var2);
                    var gradient3 = executor.GetGradient(var3);
                    return Tuple.Create(gradient1, gradient2, gradient3);
                };

            return Tuple.Create(forward, backward);
        }

        // write a function to create the graph
        public static Variable<T> Foo<T>(Variable<T> x, Variable<T> w, Variable<T> b)
        {
            return Dot(x, w) + b;
        }

        [Test]
        public static void Test()
        {
            // compile the graph on one context, then get the forward and backward computation delegate from the
            // returned tuple.
            var ctx = Context.GpuContext(0);
            var funcs = Compile<double, double, double, double>(ctx, Foo);
            var forward = funcs.Item1;
            var backward = funcs.Item2;

            // create host arrays
            var m = 100;
            var k = 90;
            var n = 80;
            var x = new double[m, k];
            var w = new double[k, n];
            var b = new double[n];

            // randomly set the host arrays
            var rng = new Random(42);
            AleaTKUtil.Common.UniformRandomArray(x, rng);
            AleaTKUtil.Common.UniformRandomArray(w, rng);
            AleaTKUtil.Common.UniformRandomArray(b, rng);

            // you can calc the output
            var y = forward(x.AsTensor(), w.AsTensor(), b.AsTensor());
            //y.Print();

            // fake some gradient
            var dy = new double[m, n];
            AleaTKUtil.Common.UniformRandomArray(dy, rng);

            // calc the gradients, they are in a tuple
            var gradients = backward(dy.AsTensor());
            var dx = gradients.Item1;
            var dw = gradients.Item2;
            var db = gradients.Item3;

            // the following code is just to verify the gradients with finite difference.
            var varX = Variable<double>();
            var varW = Variable<double>();
            var varB = Variable<double>();
            var varY = Foo(varX, varW, varB);
            var exe = new Executor(ctx, varY);
            exe.AssignTensor(varX, x.AsTensor());
            exe.AssignTensor(varW, w.AsTensor());
            exe.AssignTensor(varB, b.AsTensor());
            exe.AssignGradientDirectly(varY, dy.AsTensor());
            var bump = 1e-7;

            var dx_fd = GradientChecker.FiniteDifferenceGradient(exe, varX, bump: bump);
            //dx.Print();
            //dx_fd.Print();
            AleaTKUtil.Common.AreClose(dx_fd.ToArray2D(), dx.ToArray2D(), 1e-6);

            var dw_fd = GradientChecker.FiniteDifferenceGradient(exe, varW, bump: bump);
            //dw.Print();
            //dw_fd.Print();
            AleaTKUtil.Common.AreClose(dw_fd.ToArray2D(), dw.ToArray2D(), 1e-6);

            var db_fd = GradientChecker.FiniteDifferenceGradient(exe, varB, bump: bump);
            //db.Print();
            //db_fd.Print();
            AleaTKUtil.Common.AreClose(db_fd.ToArray(), db.ToArray(), 1e-5);
        }
    }
}
