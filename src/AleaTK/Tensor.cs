using System;
using System.Collections.Generic;
using System.Linq;
using Alea;

namespace AleaTK
{
    public class Tensor
    {
        public Tensor(Device device, BufferMemory memory, Layout layout, IntPtr ptr)
        {
            Device = device;
            Memory = memory;
            Layout = layout;
            Ptr = ptr;
        }

        public Device Device { get; }

        public BufferMemory Memory { get; }

        public Layout Layout { get; }

        public IntPtr Ptr { get; }

        public Tensor<T> Cast<T>()
        {
            var ptr = new deviceptr<T>(Ptr);
            var buffer = new Buffer<T>(Device, Memory, Layout, ptr);
            return new Tensor<T>(buffer);
        }
    }

    public class Tensor<TValue> : RExpr<TValue>, ILValue<TValue>, IRValue<TValue>
    {
        public Tensor(Buffer<TValue> buffer)
        {
            DataType = typeof (TValue);
            Layout = buffer.Layout;
            Shape = Layout.Shape;
            OpCode = OpCodes.Tensor;
            Device = buffer.Device;
            Memory = buffer.Memory;
            Buffer = buffer;
            BufferReader = buffer.CreateBufferReader();
        }

        public override Type DataType { get; }

        public override Shape Shape { get; }

        public Device Device { get; }

        public Layout Layout { get; }

        public bool IsRValue => true;

        public bool IsLValue => true;

        public BufferMemory Memory { get; }

        public Buffer<TValue> Buffer { get; }

        public BufferReader<TValue> BufferReader { get; }

        public IValue<TU> Cast<TU>()
        {
            Util.EnsureEqual(DataType, typeof (TU));
            return (IValue<TU>) this;
        }

        public ILValue<TValue> ToLValue()
        {
            return this;
        }

        public IRValue<TValue> ToRValue()
        {
            return this;
        }

        public override void Prepare(Assignment assignment)
        {
            assignment.SpecifyOutput(this, this);
        }

        protected override IRValue<TValue> GenerateRValue(Assignment assignment)
        {
            return this;
        }

        public Tensor<TValue> Slice(params Range[] ranges)
        {
            Util.EnsureTrue(Shape.Rank >= ranges.Length);

            if (Shape.Rank > ranges.Length)
            {
                var newRanges = new List<Range>();
                for (var i = 0; i < Shape.Rank; ++i)
                {
                    if (i < ranges.Length)
                    {
                        newRanges.Add(ranges[i]);
                    }
                    else
                    {
                        newRanges.Add(-1);
                    }
                }
                ranges = newRanges.ToArray();
            }

            Util.EnsureTrue(Shape.Rank == ranges.Length);

            var rank = Shape.Rank;
            var newShape = new long[rank];
            var newStrides = new long[rank];

            var ptr = Buffer.Ptr;
            for (var i = 0; i < rank; ++i)
            {
                var range = ranges[i];
                var end = range.End >= 0 ? range.End : Shape[i];
                newShape[i] = (end - range.Begin)/range.Step;
                newStrides[i] = Layout.Strides[i]*range.Step;
                ptr = ptr.LongPtr(Layout.Strides[i]*range.Begin);
            }

            var shape = new Shape(newShape);
            var strides = new Strides(newStrides);
            var layout = new Layout(shape, strides);
            var memory = Buffer.Memory;
            var buffer = new Buffer<TValue>(Device, memory, layout, ptr);
            return new Tensor<TValue>(buffer);
        }

        public Tensor<TValue> T
        {
            get
            {
                var shape = new Shape(Layout.Shape.Reverse().ToArray());
                var strides = new Strides(Layout.Strides.Reverse().ToArray());
                var layout = new Layout(shape, strides);
                var memory = Buffer.Memory;
                var buffer = new Buffer<TValue>(Device, memory, layout, Buffer.Ptr);
                return new Tensor<TValue>(buffer);
            }
        }

        public Tensor<TValue> Reshape(params long[] dims)
        {
            // -1 means calc the shape, but only one -1 allowed.
            var numNegOne = dims.Select(x => x < 0 ? 1 : 0).Sum();
            Util.EnsureTrue(numNegOne == 0 || numNegOne == 1);

            Shape newShape;
            if (numNegOne == 0)
            {
                var shape = new Shape(dims);
                // length must match old one
                Util.EnsureEqual(Shape.Length, shape.Length);
                newShape = shape;
            }
            else
            {
                var remainLength = dims.Select(x => x >= 0 ? x : 1L).Aggregate(ScalarOps.Mul);
                for (var i = 0; i < dims.Length; ++i)
                {
                    if (dims[i] < 0)
                    {
                        dims[i] = Shape.Length/remainLength;
                        break;
                    }
                }
                // check if it is multiply correct
                var shape = new Shape(dims);
                Util.EnsureEqual(Shape.Length, shape.Length);
                newShape = shape;
            }

            if (Layout.IsInnerChangeMostFullyPacked)
            {
                var buffer = new Buffer<TValue>(Device, Memory, new Layout(newShape), Buffer.Ptr);
                return new Tensor<TValue>(buffer);
            }

            throw new NotImplementedException();
        }

        public Tensor ToTensor()
        {
            return new Tensor(Device, Memory, Layout, Buffer.Ptr.Handle);
        }
    }

    public class TensorReader<T> : IRValue<T>
    {
        public TensorReader(Device device, Layout layout, Func<long, T> rawReader)
        {
            Device = device;
            Layout = layout;
            BufferReader = new BufferReader<T>(device, layout, rawReader);
        }

        public Device Device { get; }

        public Layout Layout { get; }

        public BufferMemory Memory
        {
            get { throw new InvalidOperationException("RValue doesn't have memory."); }
        }

        public bool IsRValue => true;

        public bool IsLValue => false;

        public BufferReader<T> BufferReader { get; }

        public IValue<TU> Cast<TU>()
        {
            Util.EnsureEqual(typeof(T), typeof(TU));
            return (IValue<TU>)this;
        }

        public ILValue<T> ToLValue()
        {
            throw new InvalidOperationException("TensorReader cannot be transform to LValue.");
        }

        public IRValue<T> ToRValue()
        {
            return this;
        }
    }
}
