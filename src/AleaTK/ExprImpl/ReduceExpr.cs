using System;
using System.Linq;
using Alea;
using Alea.Parallel;
using Alea.Parallel.Device;

namespace AleaTK.ExprImpl
{
    public class ReduceExpr<T> : LExpr<T>
    {
        public ReduceExpr(Expr<T> input, Func<T, T, T> reduction, bool keepDims, int[] reductionIndice, string opCode = OpCodes.Reduce)
        {
            if (reductionIndice == null || reductionIndice.Length == 0)
            {
                reductionIndice = Enumerable.Range(0, input.Shape.Rank).ToArray();
            }

            var compactReductionIndices = reductionIndice.Distinct().ToList();
            compactReductionIndices.Sort();

            Util.EnsureTrue(compactReductionIndices.Count <= input.Shape.Rank);
            Util.EnsureTrue(compactReductionIndices.All(dim => dim < input.Shape.Rank));

            if (keepDims)
            {
                var newDims = input.Shape.ToArray();
                foreach (var dim in reductionIndice)
                {
                    newDims[dim] = 1;
                }
                Shape = new Shape(newDims);
            }
            else
            {
                var newDims = input.Shape.ToList();
                foreach (var dim in reductionIndice)
                {
                    newDims[dim] = -1;
                }
                newDims.RemoveAll(l => l < 0);
                Shape = new Shape(newDims.ToArray());
            }

            OpCode = opCode;
            Input = input;
            Reduction = reduction;
            ReductionIndices = compactReductionIndices.ToArray();
            AddOperand(input);
        }

        public Expr<T> Input { get; }

        public Func<T, T, T> Reduction { get; }

        public int[] ReductionIndices { get; }

        public override Shape Shape { get; }

        protected override bool Execute(Assignment assignment, ILValue<T> output)
        {
            var input = assignment.GetInput(Input).ToRValue();
            var reduce = Reduction;
            var reductionIndices = ReductionIndices;
            var shape = Shape;

            // unit stride case and reduction is full reduction (to scalar)
            if (input.Layout.IsFullyUnitStride && shape.Rank == 0)
            {
                var length = input.Layout.Shape.Length;
                Util.EnsureTrue(length > 0L);
                var read = input.BufferReader.GetFlatReader1();
                var write = output.Buffer.FlatWriter1;

                if (assignment.Context.Type == ContextType.Gpu)
                {
                    var stream = assignment.Context.ToGpuContext().Stream;
                    var numItems = (int)length;
                    Func<int, T> sourceOp = i => read(i);
                    Action<int, T> outputOp = (i, value) => write(i, value);

                    // first pass, get the size of temp memory
                    var tempStorageSize = 0;
                    DeviceReduce.Reduce(stream, new deviceptr<byte>(), ref tempStorageSize, numItems, sourceOp, outputOp, reduce);

                    // now allocate temp memory
                    using (var tempMemoryRcpt = assignment.Context.Device.ToGpuDevice().MemoryRepository.Acquire<byte>(tempStorageSize))
                    {
                        DeviceReduce.Reduce(stream, tempMemoryRcpt.Memory.Ptr, ref tempStorageSize, numItems, sourceOp, outputOp, reduce);
                    }

                    return true;
                }

                if (assignment.Context.Type == ContextType.Cpu)
                {
                    var acc = read(0L);
                    for (var i = 1L; i < length; ++i) acc = reduce(acc, read(i));
                    write(0L, acc);
                    return true;
                }
            }

            // currently we only support matrix partial reduce, need TODO to fix this with more generic cases
            if (input.Layout.IsFullyUnitStride && input.Layout.Rank == 2 && reductionIndices.Length == 1)
            {
                var rows = input.Layout.Shape[0];
                var cols = input.Layout.Shape[1];
                var read = input.BufferReader.GetReader2();
                var write = output.Buffer.FlatWriter1;

                if (reductionIndices[0] == 1)
                {
                    if (assignment.Context.Type == ContextType.Gpu)
                    {
                        var stream = assignment.Context.ToGpuContext().Stream;
                        var numSegments = (int)rows;
                        var numItems = (int)cols;
                        Func<int, int, T> sourceOp = (i, j) => read(i, j);
                        Action<int, int, T> outputOp = (i, _, value) => write(i, value);

                        // first pass, get the size of temp memory
                        var tempStorageSize = 0;
                        DeviceReduce.Reduce(stream, new deviceptr<byte>(), ref tempStorageSize, numSegments, numItems, sourceOp, outputOp, reduce);

                        // now allocate temp memory
                        // TODO: move to assigmnet, manage the temp memory
                        using (var tempMemoryRcpt = assignment.Context.Device.ToGpuDevice().MemoryRepository.Acquire<byte>(tempStorageSize))
                        {
                            DeviceReduce.Reduce(stream, tempMemoryRcpt.Memory.Ptr, ref tempStorageSize, numSegments, numItems, sourceOp, outputOp, reduce);
                        }

                        return true;
                    }

                    if (assignment.Context.Type == ContextType.Cpu)
                    {
                        for (var row = 0L; row < rows; ++row)
                        {
                            var acc = read(row, 0L);
                            for (var col = 1L; col < cols; ++col)
                            {
                                acc = reduce(acc, read(row, col));
                            }
                            write(row, acc);
                        }
                        return true;
                    }
                }

                if (reductionIndices[0] == 0)
                {
                    if (assignment.Context.Type == ContextType.Gpu)
                    {
                        var stream = assignment.Context.ToGpuContext().Stream;

                        // This is a quick fix, it is not good performance
                        stream.For(0L, cols, col =>
                        {
                            var acc = read(0L, col);
                            for (var row = 1L; row < rows; ++row)
                            {
                                acc = reduce(acc, read(row, col));
                            }
                            write(col, acc);
                        });

                        return true;
                    }

                    if (assignment.Context.Type == ContextType.Cpu)
                    {
                        for (var col = 0L; col < cols; ++col)
                        {
                            var acc = read(0L, col);
                            for (var row = 1L; row < rows; ++row)
                            {
                                acc = reduce(acc, read(row, col));
                            }
                            write(col, acc);
                        }
                        return true;
                    }
                }
            }

            return false;
        }
    }
}