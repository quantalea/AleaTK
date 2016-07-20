using System;
using Alea;
using Alea.cuRAND;

namespace AleaTK.ExprImpl
{
    public enum PseudoRandomType
    {
        Default = 0
    }

    public class PseudoRandomExpr<T> : LExpr<T>
    {
        public PseudoRandomExpr(Shape shape, PseudoRandomType type, Distribution distribution, ulong seed, ulong offset, string opCode = OpCodes.Random)
        {
            OpCode = opCode;
            Shape = shape;
            Seed = seed;
            Distribution = distribution;
            Offset = offset;
            Type = type;
        }

        private RngType CuRandType
        {
            get
            {
                switch (Type)
                {
                    case PseudoRandomType.Default:
                        return RngType.PSEUDO_DEFAULT;

                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }
        }

        public PseudoRandomType Type { get; }

        public Distribution Distribution { get; }

        public ulong Seed { get; }

        public ulong Offset { get; }

        public override Shape Shape { get; }

        protected override bool Execute(Assignment assignment, ILValue<T> output)
        {
            if (assignment.Context.Type == ContextType.Gpu)
            {
                var stream = assignment.Context.ToGpuContext().Stream;
                var gpu = stream.Gpu;

                // TODO: manage rng, less creation
                using (var rng = Generator.CreateGpu(gpu, CuRandType))
                {
                    rng.SetStream(stream);
                    rng.SetPseudoRandomGeneratorSeed(Seed);
                    rng.SetGeneratorOrdering(Ordering.PSEUDO_DEFAULT);
                    rng.SetGeneratorOffset(Offset);

                    if (typeof(T) == typeof(double))
                    {
                        var ptr = output.Buffer.Ptr.Reinterpret<double>();

                        if (Distribution is UniformDistribution)
                        {
                            rng.GenerateUniform(ptr, (ulong) output.Layout.Shape.Length);
                            return true;
                        }

                        if (Distribution is NormalDistribution)
                        {
                            var dist = Distribution as NormalDistribution;
                            rng.GenerateNormal(ptr, (ulong) output.Layout.Shape.Length, dist.Mean, dist.Stddev);
                            return true;
                        }

                        throw new InvalidOperationException();
                    }

                    if (typeof(T) == typeof(float))
                    {
                        var ptr = output.Buffer.Ptr.Reinterpret<float>();

                        if (Distribution is UniformDistribution)
                        {
                            rng.GenerateUniform(ptr, (ulong)output.Layout.Shape.Length);
                            return true;
                        }

                        if (Distribution is NormalDistribution)
                        {
                            var dist = Distribution as NormalDistribution;
                            rng.GenerateNormal(ptr, (ulong)output.Layout.Shape.Length, (float)dist.Mean, (float)dist.Stddev);
                            return true;
                        }

                        throw new InvalidOperationException();
                    }

                    if (typeof(T) == typeof(double2))
                    {
                        var ptr = output.Buffer.Ptr.Reinterpret<double>();

                        if (Distribution is UniformDistribution)
                        {
                            rng.GenerateUniform(ptr, (ulong)output.Layout.Shape.Length * 2UL);
                            return true;
                        }

                        if (Distribution is NormalDistribution)
                        {
                            var dist = Distribution as NormalDistribution;
                            rng.GenerateNormal(ptr, (ulong)output.Layout.Shape.Length * 2UL, dist.Mean, dist.Stddev);
                            return true;
                        }

                        throw new InvalidOperationException();
                    }

                    if (typeof(T) == typeof(uint) || typeof(T) == typeof(int))
                    {
                        var ptr = output.Buffer.Ptr.Reinterpret<uint>();

                        if (Distribution is UniformDistribution)
                        {
                            rng.Generate(ptr, (ulong)output.Layout.Shape.Length);
                            return true;
                        }

                        throw new InvalidOperationException();
                    }
                }
            }

            if (assignment.Context.Type == ContextType.Cpu)
            {
                using (var rng = Generator.CreateCpu(CuRandType))
                {
                    rng.SetPseudoRandomGeneratorSeed(Seed);
                    rng.SetGeneratorOrdering(Ordering.PSEUDO_DEFAULT);
                    rng.SetGeneratorOffset(Offset);

                    if (typeof(T) == typeof(double))
                    {
                        var ptr = output.Buffer.Ptr.Reinterpret<double>();

                        if (Distribution is UniformDistribution)
                        {
                            rng.GenerateUniform(ptr, (ulong)output.Layout.Shape.Length);
                            return true;
                        }

                        if (Distribution is NormalDistribution)
                        {
                            var dist = Distribution as NormalDistribution;
                            rng.GenerateNormal(ptr, (ulong)output.Layout.Shape.Length, dist.Mean, dist.Stddev);
                            return true;
                        }

                        throw new InvalidOperationException();
                    }

                    if (typeof(T) == typeof(float))
                    {
                        var ptr = output.Buffer.Ptr.Reinterpret<float>();

                        if (Distribution is UniformDistribution)
                        {
                            rng.GenerateUniform(ptr, (ulong)output.Layout.Shape.Length);
                            return true;
                        }

                        if (Distribution is NormalDistribution)
                        {
                            var dist = Distribution as NormalDistribution;
                            rng.GenerateNormal(ptr, (ulong)output.Layout.Shape.Length, (float)dist.Mean, (float)dist.Stddev);
                            return true;
                        }

                        throw new InvalidOperationException();
                    }

                    if (typeof(T) == typeof(double2))
                    {
                        var ptr = output.Buffer.Ptr.Reinterpret<double>();

                        if (Distribution is UniformDistribution)
                        {
                            rng.GenerateUniform(ptr, (ulong)output.Layout.Shape.Length * 2UL);
                            return true;
                        }

                        if (Distribution is NormalDistribution)
                        {
                            var dist = Distribution as NormalDistribution;
                            rng.GenerateNormal(ptr, (ulong)output.Layout.Shape.Length * 2UL, dist.Mean, dist.Stddev);
                            return true;
                        }

                        throw new InvalidOperationException();
                    }

                    if (typeof(T) == typeof(uint) || typeof(T) == typeof(int))
                    {
                        var ptr = output.Buffer.Ptr.Reinterpret<uint>();

                        if (Distribution is UniformDistribution)
                        {
                            rng.Generate(ptr, (ulong)output.Layout.Shape.Length);
                            return true;
                        }

                        throw new InvalidOperationException();
                    }
                }
            }

            return false;
        }
    }
}