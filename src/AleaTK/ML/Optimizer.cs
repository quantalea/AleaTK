using System;
using System.Collections.Generic;
using static AleaTK.Library;

namespace AleaTK.ML
{
    public abstract class GradientClipper
    {
        public abstract void Clip(Executor executor);
    }

    public class NoGradientClipper : GradientClipper
    {
        public override void Clip(Executor executor)
        {
        }
    }

    public class GlobalNormGradientClipper : GradientClipper
    {
        public readonly double ClipNorm;

        public GlobalNormGradientClipper(double clipNorm)
        {
            Util.EnsureTrue(clipNorm > 0.0, "clip norm should > 0");
            ClipNorm = clipNorm;
        }

        public override void Clip(Executor executor)
        {
            var l2normSqure = 0.0;

            foreach (var data in executor.Data)
            {
                if (data.Variable.Type == VariableType.Parameter)
                {
                    if (data.Variable.DataType == typeof(float))
                    {
                        var variable = (Variable<float>)data.Variable;
                        var gradient = executor.GetGradient(variable);
                        l2normSqure += executor.Context.Eval(AleaTK.Library.ReduceSum(gradient * gradient)).ToScalar();
                    }
                    else if (data.Variable.DataType == typeof(double))
                    {
                        var variable = (Variable<double>)data.Variable;
                        var gradient = executor.GetGradient(variable);
                        l2normSqure += executor.Context.Eval(AleaTK.Library.ReduceSum(gradient * gradient)).ToScalar();
                    }
                    else
                    {
                        throw new InvalidOperationException($"Unsupported gradient type {data.Variable.DataType}.");
                    }
                }
            }

            var l2norm = Math.Sqrt(l2normSqure);

            if (l2norm > ClipNorm)
            {
                foreach (var data in executor.Data)
                {
                    if (data.Variable.Type == VariableType.Parameter)
                    {
                        if (data.Variable.DataType == typeof(float))
                        {
                            var variable = (Variable<float>)data.Variable;
                            var gradient = executor.GetGradient(variable);
                            executor.Context.Assign(gradient, gradient * ClipNorm.AsScalar<float>() / l2norm.AsScalar<float>());
                        }
                        else if (data.Variable.DataType == typeof(double))
                        {
                            var variable = (Variable<double>)data.Variable;
                            var gradient = executor.GetGradient(variable);
                            executor.Context.Assign(gradient, gradient * ClipNorm.AsScalar<double>() / l2norm.AsScalar<double>());
                        }
                        else
                        {
                            throw new InvalidOperationException($"Unsupported gradient type {data.Variable.DataType}.");
                        }
                    }
                }
            }
        }
    }

    public class NormGradientClipper : GradientClipper
    {
        public readonly double ClipNorm;

        public NormGradientClipper(double clipNorm)
        {
            Util.EnsureTrue(clipNorm > 0.0, "clip norm should > 0");
            ClipNorm = clipNorm;
        }

        public void Clip(Executor executor, Variable var)
        {
            if (var.DataType == typeof(float))
            {
                var variable = (Variable<float>)var;
                var gradient = executor.GetGradient(variable);
                var l2normSqure = executor.Context.Eval(AleaTK.Library.ReduceSum(gradient * gradient)).ToScalar();
                var l2norm = Math.Sqrt(l2normSqure);

                if (l2norm > ClipNorm)
                {
                    executor.Context.Assign(gradient, gradient * ClipNorm.AsScalar<float>() / l2norm.AsScalar<float>());
                }

                return;
            }

            if (var.DataType == typeof(double))
            {
                var variable = (Variable<double>)var;
                var gradient = executor.GetGradient(variable);
                var l2normSqure = executor.Context.Eval(AleaTK.Library.ReduceSum(gradient * gradient)).ToScalar();
                var l2norm = Math.Sqrt(l2normSqure);

                if (l2norm > ClipNorm)
                {
                    executor.Context.Assign(gradient, gradient * ClipNorm.AsScalar<double>() / l2norm.AsScalar<double>());
                }

                return;
            }

            throw new InvalidOperationException($"Unsupported gradient type {var.DataType}.");
        }

        public override void Clip(Executor executor)
        {
            foreach (var data in executor.Data)
            {
                if (data.Variable.Type == VariableType.Parameter)
                {
                    Clip(executor, data.Variable);
                }
            }
        }
    }

    public abstract class Optimizer : Executor
    {
        protected Optimizer(Context ctx, Variable output) : base(ctx, output)
        {
        }

        public abstract void Optimize();
    }

    public class GradientDescentOptimizer : Optimizer
    {
        public GradientDescentOptimizer(Context ctx, Variable output, double learningRate) : base(ctx, output)
        {
            LearningRate = learningRate;
            GradientClipper = new NoGradientClipper();
        }

        public GradientDescentOptimizer(Context ctx, Variable output, double learningRate, GradientClipper clipper) : base(ctx, output)
        {
            LearningRate = learningRate;
            GradientClipper = clipper;
        }

        public double LearningRate { get; }

        public GradientClipper GradientClipper { get; }

        public override void Optimize()
        {
            Optimize(LearningRate);
        }

        public void Optimize(double learningRate)
        {
            GradientClipper.Clip(this);

            foreach (var data in Data)
            {
                if (data.Variable.Type == VariableType.Parameter)
                {
                    var w = data.TensorAsExpr;
                    var g = data.GradientAsExpr;
                    Context.Assign(data.TensorAsValue, w - learningRate.AsScalar(w.DataType) * g);
                }
            }
        }
    }

    public class RMSpropOptimizer : Optimizer
    {
        private readonly Dictionary<Variable, Tensor> _weights = new Dictionary<Variable, Tensor>();

        public RMSpropOptimizer(Context ctx, Variable output, double learningRate, double rho, double epsilon) : base(ctx, output)
        {
            LearningRate = learningRate;
            Rho = rho;
            Epsilon = epsilon;
            GradientClipper = new NoGradientClipper();
        }

        public RMSpropOptimizer(Context ctx, Variable output, double learningRate, double rho, double epsilon, GradientClipper clipper) : base(ctx, output)
        {
            LearningRate = learningRate;
            Rho = rho;
            Epsilon = epsilon;
            GradientClipper = clipper;
        }

        public double LearningRate { get; }

        public double Rho { get; }

        public double Epsilon { get; }

        public GradientClipper GradientClipper { get; }

        public override void Initalize()
        {
            base.Initalize();

            foreach (var data in Data)
            {
                if (data.Variable.Type == VariableType.Parameter)
                {
                    Tensor tensor = null;
                    var shape = Shape.Create(data.Tensor.Layout.Shape.AsArray);
                    var layout = new Layout(shape);
                    data.Variable.GetOrAllocate(Context.Device, layout, shape.Length, ref tensor);

                    if (data.Variable.DataType == typeof (float))
                    {
                        Context.Assign(tensor.Cast<float>(), Fill(shape, 0.0f));
                    }
                    else if (data.Variable.DataType == typeof(double))
                    {
                        Context.Assign(tensor.Cast<double>(), Fill(shape, 0.0));
                    }
                    else
                    {
                        throw new InvalidOperationException("Wrong data type");
                    }

                    _weights.Add(data.Variable, tensor);
                }
            }
        }

        public override void Optimize()
        {
            GradientClipper.Clip(this);

            foreach (var data in Data)
            {
                if (data.Variable.Type == VariableType.Parameter)
                {
                    if (data.Variable.DataType == typeof (float))
                    {
                        var a = _weights[data.Variable].Cast<float>();
                        var p = data.Tensor.Cast<float>();
                        var g = data.Gradient.Cast<float>();
                        Context.Assign(a, Rho.AsScalar<float>()*a + (1.0.AsScalar<float>() - Rho.AsScalar<float>())*g*g);
                        Context.Assign(p, p - LearningRate.AsScalar<float>()*g/(Sqrt(a) + Epsilon.AsScalar<float>()));
                    }
                    else if (data.Variable.DataType == typeof(double))
                    {
                        var a = _weights[data.Variable].Cast<double>();
                        var p = data.Tensor.Cast<double>();
                        var g = data.Gradient.Cast<double>();
                        Context.Assign(a, Rho.AsScalar<double>() * a + (1.0.AsScalar<double>() - Rho.AsScalar<double>()) * g * g);
                        Context.Assign(p, p - LearningRate.AsScalar<double>() * g / (Sqrt(a) + Epsilon.AsScalar<double>()));
                    }
                    else
                    {
                        throw new InvalidOperationException();
                    }
                }
            }
        }
    }
}