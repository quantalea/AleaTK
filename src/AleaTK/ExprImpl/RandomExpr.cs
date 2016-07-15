using System;
using Alea;
using Alea.cuRAND;

namespace AleaTK.ExprImpl
{
    public enum RandomDistribution
    {
        Uniform = 0
    }

    public enum PseudoRandomType
    {
        Default = 0
    }

    public class PseudoRandomExpr<T> : LExpr<T>
    {
        public PseudoRandomExpr(Shape shape, PseudoRandomType type, RandomDistribution randomDistribution, ulong seed, ulong offset, string opCode = OpCodes.Random)
        {
            OpCode = opCode;
            Shape = shape;
            Seed = seed;
            RandomDistribution = randomDistribution;
            Offset = offset;
            Type = type;
        }

        public PseudoRandomExpr(PseudoRandomType type, RandomDistribution randomDistribution, ulong seed, ulong offset, string opCode = OpCodes.Random)
        {
            OpCode = opCode;
            Shape = null;
            Seed = seed;
            RandomDistribution = randomDistribution;
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

        public RandomDistribution RandomDistribution { get; }

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
                        switch (RandomDistribution)
                        {
                            case RandomDistribution.Uniform:
                                rng.GenerateUniform(ptr, (ulong)output.Layout.Shape.Length);
                                break;
                            default:
                                throw new ArgumentOutOfRangeException();
                        }
                        return true;
                    }

                    if (typeof(T) == typeof(float))
                    {
                        var ptr = output.Buffer.Ptr.Reinterpret<float>();
                        switch (RandomDistribution)
                        {
                            case RandomDistribution.Uniform:
                                rng.GenerateUniform(ptr, (ulong)output.Layout.Shape.Length);
                                break;
                            default:
                                throw new ArgumentOutOfRangeException();
                        }
                        return true;
                    }

                    if (typeof(T) == typeof(double2))
                    {
                        var ptr = output.Buffer.Ptr.Reinterpret<double>();
                        switch (RandomDistribution)
                        {
                            case RandomDistribution.Uniform:
                                rng.GenerateUniform(ptr, (ulong)output.Layout.Shape.Length * 2UL);
                                break;
                            default:
                                throw new ArgumentOutOfRangeException();
                        }
                        return true;
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
                        switch (RandomDistribution)
                        {
                            case RandomDistribution.Uniform:
                                rng.GenerateUniform(ptr, (ulong)output.Layout.Shape.Length);
                                break;
                            default:
                                throw new ArgumentOutOfRangeException();
                        }
                        return true;
                    }

                    if (typeof(T) == typeof(float))
                    {
                        var ptr = output.Buffer.Ptr.Reinterpret<float>();
                        switch (RandomDistribution)
                        {
                            case RandomDistribution.Uniform:
                                rng.GenerateUniform(ptr, (ulong)output.Layout.Shape.Length);
                                break;
                            default:
                                throw new ArgumentOutOfRangeException();
                        }
                        return true;
                    }

                    if (typeof(T) == typeof(double2))
                    {
                        var ptr = output.Buffer.Ptr.Reinterpret<double>();
                        switch (RandomDistribution)
                        {
                            case RandomDistribution.Uniform:
                                rng.GenerateUniform(ptr, (ulong)output.Layout.Shape.Length * 2UL);
                                break;
                            default:
                                throw new ArgumentOutOfRangeException();
                        }
                        return true;
                    }
                }
            }

            return false;
        }
    }
}