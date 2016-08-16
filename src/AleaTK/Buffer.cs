using System;
using System.Linq;
using Alea;

namespace AleaTK
{
    public sealed class BufferMemory : Disposable
    {
        public BufferMemory(Memory memory, long offset, long length)
        {
            Memory = memory;
            Offset = offset;
            Length = length;
        }

        public Memory Memory { get; }

        public long Offset { get; }

        public long Length { get; }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                Memory.Dispose();
            }
            base.Dispose(disposing);
        }
    }

    public class Buffer<T> : Disposable
    {
        private readonly HostPtrAccessor<T> _hptr; 

        public Buffer(Device device, BufferMemory memory, Layout layout, deviceptr<T> ptr)
        {
            Device = device;
            Memory = memory;
            Layout = layout;
            Ptr = ptr;

            switch (Device.Type)
            {
                case DeviceType.Gpu:
                    _hptr = null;
                    var dptr = ptr;
                    RawReader = i => dptr.LongGet(i);
                    RawWriter = (i, value) => dptr.LongSet(i, value);
                    break;

                case DeviceType.Cpu:
                    var hptr = new HostPtrAccessor<T>(memory, ptr);
                    _hptr = hptr;
                    RawReader = hptr.Get;
                    RawWriter = hptr.Set;
                    break;

                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public Device Device { get; }

        public BufferMemory Memory { get; }

        public Layout Layout { get; }

        public deviceptr<T> Ptr { get; }

        public HostPtrAccessor<T> CpuPtr
        {
            get
            {
                if (_hptr != null) return _hptr;
                throw new InvalidOperationException("This buffer is not CPU buffer.");
            }
        }

        public Func<long, T> RawReader { get; }

        public Action<long, T> RawWriter { get; }

        // when the layout is fully unit stride, you can get flatten 1d accessor
        public Func<long, T> GetFlatReader1(Shape broadcastShape = null)
        {
            Util.EnsureTrue(Layout.IsFullyUnitStride);
            var rawReader = RawReader;
            var stride = Layout.FullyUnitStride;

            if (stride == 0L) return _ => rawReader(0L);

            if (broadcastShape == null || Layout.Shape.SequenceEqual(broadcastShape))
            {
                return stride == 1L ? rawReader : i => rawReader(i*stride);
            }

            var shape = Layout.Shape;
            Util.EnsureTrue(broadcastShape.Rank >= shape.Rank);

            // all inner broadcasting
            if (broadcastShape.Skip(broadcastShape.Rank - shape.Rank).Zip(shape, (l1, l2) => l1 == l2).All(pred => pred))
            {
                var length = shape.Length;
                if (stride == 1L) return i => rawReader(i%length);
                return i => rawReader((i%length)*stride);
            }

            // non-all-inner broadcasting
            if (broadcastShape.Rank == 2)
            {
                var dstCols = broadcastShape[1];
                // ReSharper disable once PossibleNullReferenceException
                if (stride == 1L) return i => rawReader(i / dstCols);
                // ReSharper disable once PossibleNullReferenceException
                return i => rawReader((i / dstCols) * stride);
            }

            throw new NotImplementedException();
        }

        public Action<long, T> FlatWriter1
        {
            get
            {
                var rawWriter = RawWriter;
                var stride = Layout.FullyUnitStride;
                switch (stride)
                {
                    case 1L: return rawWriter;
                    case 0L: return (_, value) => rawWriter(0L, value);
                    default: return (i, value) => rawWriter(i*stride, value);
                }
            }
        }

        public Func<long, long, T> GetReader2(Shape broadcastShape = null)
        {
            Util.EnsureTrue(Layout.Rank == 2);
            var stride0 = Layout.Strides[0];
            var stride1 = Layout.Strides[1];
            var rawReader = RawReader;

            if (broadcastShape == null || broadcastShape.SequenceEqual(Layout.Shape))
            {
                return (i0, i1) => rawReader(i0 * stride0 + i1 * stride1);
            }

            throw new NotImplementedException();
        }

        public Action<long, long, T> Writer2
        {
            get
            {
                Util.EnsureTrue(Layout.Rank == 2);
                var stride0 = Layout.Strides[0];
                var stride1 = Layout.Strides[1];
                var rawWriter = RawWriter;
                return (i0, i1, value) => rawWriter(i0 * stride0 + i1 * stride1, value);
            }
        }

        public Action<long, long, long, T> Writer3
        {
            get
            {
                Util.EnsureTrue(Layout.Rank == 3);
                var stride0 = Layout.Strides[0];
                var stride1 = Layout.Strides[1];
                var stride2 = Layout.Strides[2];
                var rawWriter = RawWriter;
                return (i0, i1, i2, value) => rawWriter(i0 * stride0 + i1 * stride1 + i2 * stride2, value);
            }
        }

        public BufferReader<T> CreateBufferReader()
        {
            return new BufferReader<T>(Device, Layout, RawReader);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                Memory.Dispose();
            }
            base.Dispose(disposing);
        }
    }

    public class BufferReader<T>
    {
        public BufferReader(Device device, Layout layout, Func<long, T> rawReader)
        {
            Device = device;
            Shape = layout.Shape;
            Layout = layout;
            RawReader = rawReader;
        }

        public BufferReader(Device device, Shape shape)
        {
            Device = device;
            Shape = shape;
        } 

        public Device Device { get; }

        public Shape Shape { get; }

        public Layout Layout { get; }

        public Func<long, T> RawReader { get; }

        public Func<long, T> GetFlatReader1(Shape broadcastShape = null)
        {
            Util.EnsureTrue(RawReader != null);
            Util.EnsureTrue(Layout != null && Layout.IsFullyUnitStride);

            var rawReader = RawReader;
            // ReSharper disable once PossibleNullReferenceException
            var stride = Layout.FullyUnitStride;

            // ReSharper disable once PossibleNullReferenceException
            if (stride == 0L) return _ => rawReader(0L);

            if (broadcastShape == null || Layout.Shape.SequenceEqual(broadcastShape))
            {
                // ReSharper disable once PossibleNullReferenceException
                return stride == 1L ? rawReader : i => rawReader(i * stride);
            }

            var shape = Layout.Shape;
            Util.EnsureTrue(broadcastShape.Rank >= shape.Rank);

            if (shape.Length == 1)
            {
                return i => rawReader(0);
            }

            // all inner broadcasting
            var firstNonOneIndex = shape.FirstIndex(l => l != 1L);
            var innerShape = firstNonOneIndex < shape.Rank - 1
                ? Shape.Create(shape.Skip(firstNonOneIndex + 1).ToArray())
                : shape;
            if (broadcastShape.Skip(broadcastShape.Rank - innerShape.Rank).Zip(shape, (l1, l2) => l1 == l2).All(pred => pred))
            {
                var length = innerShape.Length;
                // ReSharper disable once PossibleNullReferenceException
                if (stride == 1L) return i => rawReader(i % length);
                // ReSharper disable once PossibleNullReferenceException
                return i => rawReader((i % length) * stride);
            }

            var strides = Layout.Strides;

            // non-all-inner broadcasting
            if (broadcastShape.Rank == 2)
            {
                var dstCols = broadcastShape[1];
                // ReSharper disable once PossibleNullReferenceException
                if (stride == 1L) return i => rawReader(i/dstCols);
                // ReSharper disable once PossibleNullReferenceException
                return i => rawReader((i/dstCols)*stride);
            }

            if (broadcastShape.Rank == 3)
            {
                // extend shapes to that rank
                var extenededShape = new Shape(Enumerable.Repeat(1L, broadcastShape.Rank - shape.Rank).Concat(shape).ToArray());
                var length1 = broadcastShape[1];
                var length2 = broadcastShape[2];
                var stride0 = extenededShape[0] == broadcastShape[0] ? extenededShape[1]*extenededShape[2]*stride : 0;
                var stride1 = extenededShape[1] == broadcastShape[1] ? extenededShape[2] * stride : 0;
                var stride2 = extenededShape[2] == broadcastShape[2] ? stride : 0;
                return i =>
                {
                    var i0 = i/(length1*length2);
                    var i1 = i%(length1*length2)/length2;
                    var i2 = i%(length1*length2)%length2;
                    var idx = i0*stride0 + i1*stride1 + i2*stride2;
                    return rawReader(idx);
                };
            }

            throw new NotImplementedException($"{shape} => {broadcastShape}");
        }

        public Func<long, long, T> GetReader2(Shape broadcastShape = null)
        {
            Util.EnsureTrue(Layout.Rank == 2);

            var length0 = Shape[0];
            var length1 = Shape[1];

            var stride0 = Layout.Strides[0];
            var stride1 = Layout.Strides[1];
            var rawReader = RawReader;

            if (broadcastShape == null || broadcastShape.SequenceEqual(Layout.Shape))
            {
                return (i0, i1) => rawReader(i0 * stride0 + i1 * stride1);
            }

            return (i0, i1) => rawReader((i0% length0) *stride0 + (i1% length1) *stride1);
        }

        public Func<long, long, long, T> GetReader3(Shape broadcastShape = null)
        {
            Util.EnsureTrue(Layout.Rank == 3);

            var length0 = Shape[0];
            var length1 = Shape[1];
            var length2 = Shape[2];

            var stride0 = Layout.Strides[0];
            var stride1 = Layout.Strides[1];
            var stride2 = Layout.Strides[2];
            var rawReader = RawReader;

            if (broadcastShape == null || broadcastShape.SequenceEqual(Layout.Shape))
            {
                return (i0, i1, i2) => rawReader(i0 * stride0 + i1 * stride1 + i2 * stride2);
            }

            return (i0, i1, i2) => rawReader((i0 % length0) * stride0 + (i1 % length1) * stride1 + (i2 % length2) * stride2);
        }
    }

    public static class Buffer
    {
        public static Buffer<T> Create<T>(Device device, Memory<T> memory, long offset, long length, Layout layout, deviceptr<T> ptr)
        {
            return new Buffer<T>(device, new BufferMemory(memory, offset, length), layout, ptr);
        }

        public static Buffer<T> Reference<T>(Array array, Shape shape)
        {
            Util.EnsureEqual(typeof (T), array.GetType().GetElementType());
            var device = Device.CpuDevice;
            var memory = new ManagedArrayMemory<T>(array);
            var length = array.LongLength;
            var layout = new Layout(shape);
            var ptr = memory.Ptr;
            return Create(device, memory, 0L, length, layout, ptr);
        }

        public static Buffer<T> Reference<T>(Array array)
        {
            return Reference<T>(array, Shape.GetArrayShape(array));
        }

        public static Buffer<T> Allocate<T>(Device device, Layout layout, long length)
        {
            switch (device.Type)
            {
                case DeviceType.Gpu:
                {
                    var gpu = device.ToGpuDevice().Gpu;
                    var memory = gpu.AllocateDevice<T>(length);
                    var ptr = memory.Ptr;
                    return Create(device, memory, 0L, length, layout, ptr);
                }

                case DeviceType.Cpu:
                {
                    BuilderRegistry.Initialize();
                    var array = new T[length];
                    var memory = new ManagedArrayMemory<T>(array);
                    var ptr = memory.Ptr;
                    return Create(device, memory, 0L, length, layout, ptr);
                }

                default:
                    throw new ArgumentOutOfRangeException();
            }
        }
    }
}
