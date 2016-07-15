using System;
using System.Collections.Concurrent;
using Alea;
using GpuStream = Alea.Stream;

namespace AleaTK
{
    public enum DeviceType
    {
        Cpu = 0,
        Gpu
    }

    public abstract class Device
    {
        public abstract DeviceType Type { get; }

        public void EnsureType(DeviceType targetType)
        {
            if (Type != targetType)
            {
                throw new InvalidOperationException("Device type doesn't match.");
            }
        }

        public CpuDevice ToCpuDevice()
        {
            EnsureType(DeviceType.Cpu);
            return (CpuDevice)this;
        }

        public GpuDevice ToGpuDevice()
        {
            EnsureType(DeviceType.Gpu);
            return (GpuDevice)this;
        }

        private static readonly ConcurrentDictionary<int, GpuDevice> GpuDevices =
            new ConcurrentDictionary<int, GpuDevice>();

        public static readonly Device CpuDevice = new CpuDevice();

        public static Device GpuDevice(Gpu gpu)
        {
            return GpuDevices.GetOrAdd(gpu.Device.Id, deviceId => new GpuDevice(Gpu.Get(deviceId)));
        }

        public static Device GpuDevice(int deviceId)
        {
            return GpuDevices.GetOrAdd(deviceId, _ => new GpuDevice(Gpu.Get(deviceId)));
        }
    }

    public sealed class CpuDevice : Device
    {
        internal CpuDevice()
        {
        }

        public override DeviceType Type => DeviceType.Cpu;

        public override string ToString()
        {
            return "CpuDevice";
        }
    }

    public sealed class GpuDevice : Device
    {
        internal GpuDevice(Gpu gpu)
        {
            Gpu = gpu;
            MemoryRepository = new DeviceMemoryRepository(gpu);
        }

        public override DeviceType Type => DeviceType.Gpu;

        public Gpu Gpu { get; }

        public DeviceMemoryRepository MemoryRepository { get; }

        public override string ToString()
        {
            return $"GpuDevice{Gpu}";
        }
    }

    public enum ContextType
    {
        Cpu = 0,
        Gpu
    }

    public abstract class Context : Disposable
    {
        public abstract ContextType Type { get; }

        public abstract Device Device { get; }

        public void EnsureType(ContextType targetType)
        {
            if (Type != targetType)
            {
                throw new InvalidOperationException("Context type doesn't match.");
            }
        }

        public CpuContext ToCpuContext()
        {
            EnsureType(ContextType.Cpu);
            return (CpuContext)this;
        }

        public GpuContext ToGpuContext()
        {
            EnsureType(ContextType.Gpu);
            return (GpuContext)this;
        }

        private struct GpuContextKey
        {
            private int _deviceId;
            private int _streamId;

            public GpuContextKey(int deviceId, int streamId)
            {
                _deviceId = deviceId;
                _streamId = streamId;
            }
        }

        private static readonly ConcurrentDictionary<GpuContextKey, GpuContext> GpuContexts =
            new ConcurrentDictionary<GpuContextKey, GpuContext>();

        public static readonly Context CpuContext = new CpuContext();

        public static Context GpuContext(int deviceId, int streamId = 0)
        {
            var key = new GpuContextKey(deviceId, streamId);
            return GpuContexts.GetOrAdd(key, _ => new GpuContext(Gpu.Get(deviceId)));
        }

        public static Context GpuContext(Gpu gpu, int streamId = 0)
        {
            var key = new GpuContextKey(gpu.Device.Id, streamId);
            return GpuContexts.GetOrAdd(key, _ => new GpuContext(gpu));
        }
    }

    public sealed class CpuContext : Context
    {
        public CpuContext()
        {
            Device = Device.CpuDevice;
        }

        public override ContextType Type => ContextType.Cpu;

        public override Device Device { get; }
    }

    public sealed class GpuContext : Context
    {
        public GpuContext(Gpu gpu, bool useDefaultStream = false)
        {
            Device = Device.GpuDevice(gpu);
            Gpu = gpu;
            if (useDefaultStream)
            {
                throw new NotImplementedException("TODO: add default stream support.");
            }
            else
            {
                Stream = gpu.CreateStream();
                if (Alea.cuDNN.Dnn.IsAvailable)
                {
                    Dnn = new Alea.cuDNN.Dnn(Stream);
                }
                if (Alea.cuBLAS.Blas.IsAvailable)
                {
                    Blas = new Alea.cuBLAS.Blas(Stream);
                }
            }
        }

        public override ContextType Type => ContextType.Gpu;

        public override Device Device { get; }

        public GpuStream Stream { get; }

        public Gpu Gpu { get; }

        public Alea.cuDNN.Dnn Dnn { get; }

        public Alea.cuBLAS.Blas Blas { get; }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (!Stream.IsDefault) Stream.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}