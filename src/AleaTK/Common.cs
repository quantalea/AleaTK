using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Security.Policy;
using Alea;
using Alea.cuDNN;

namespace AleaTK
{
    public static class ScalarOps
    {
        public static T Conv<T>(double v)
        {
            if (typeof (T) == typeof (double)) return (T) (object) v;
            if (typeof (T) == typeof (float)) return (T) (object) (float) v;
            throw new NotImplementedException();
        }


        #region Neg
        public static double Neg(double a)
        {
            return -a;
        }

        public static float Neg(float a)
        {
            return -a;
        }
        #endregion

        #region Add
        public static int Add(int a, int b)
        {
            return a + b;
        }

        public static long Add(long a, long b)
        {
            return a + b;
        }

        public static float Add(float a, float b)
        {
            return a + b;
        }

        public static double Add(double a, double b)
        {
            return a + b;
        }
        #endregion

        #region Sub
        public static double Sub(double a, double b)
        {
            return a - b;
        }

        public static float Sub(float a, float b)
        {
            return a - b;
        }
        #endregion

        #region Mul
        public static int Mul(int a, int b)
        {
            return a * b;
        }

        public static long Mul(long a, long b)
        {
            return a * b;
        }

        public static float Mul(float a, float b)
        {
            return a * b;
        }

        public static double Mul(double a, double b)
        {
            return a * b;
        }
        #endregion

        #region Div
        public static double Div(double a, double b)
        {
            return a / b;
        }

        public static float Div(float a, float b)
        {
            return a / b;
        }
        #endregion

        #region DivUp
        public static int DivUp(int num, int den)
        {
            return (num + den - 1) / den;
        }

        public static long DivUp(long num, long den)
        {
            return (num + den - 1) / den;
        }
        #endregion
    }

    public abstract class Distribution
    {
    }

    public sealed class UniformDistribution : Distribution
    {
    }

    public sealed class NormalDistribution : Distribution
    {
        public readonly double Mean;
        public readonly double Stddev;

        public NormalDistribution()
        {
            Mean = 0.0;
            Stddev = 1.0;
        }

        public NormalDistribution(double mean, double stddev)
        {
            Mean = mean;
            Stddev = stddev;
        }
    }

    public enum PseudoRandomType
    {
        Default = 0
    }

    public struct Range
    {
        public readonly long Begin;
        public readonly long End;
        public readonly long Step;

        private Range(long begin, long end, long step)
        {
            Begin = begin;
            End = end;
            Step = step;
        }

        //public static implicit operator Range(int idx)
        //{
        //    if (idx >= 0)
        //    {
        //        return Range.Create(idx);
        //    }

        //    return All;
        //}

        public static implicit operator Range(long idx)
        {
            if (idx >= 0)
            {
                return Range.Create(idx);
            }

            return All;
        }

        public static Range All = new Range(0, -1, 1);

        public static Range Create(long idx)
        {
            return new Range(idx, idx + 1, 1);
        }

        public static Range Create(long begin, long end)
        {
            return new Range(begin, end, 1);
        }
    }

    public abstract class Disposable : IDisposable
    {
        private bool _isDisposed = false;

        public bool IsDisposed
        {
            get
            {
                lock (this)
                {
                    return _isDisposed;
                }
            }
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                lock (this)
                {
                    if (!_isDisposed)
                    {
                        _isDisposed = true;
                    }
                }
            }
            else
            {
                if (!_isDisposed)
                {
                    _isDisposed = true;
                }
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        ~Disposable()
        {
            Dispose(false);
        }
    }

    public class DisposableRepository<T> : Disposable where T : IDisposable
    {
        public sealed class Receipt : Disposable
        {
            public Receipt(DisposableRepository<T> repository, T value)
            {
                Repository = repository;
                Value = value;
            }

            public DisposableRepository<T> Repository { get; }

            public T Value { get; }

            protected override void Dispose(bool disposing)
            {
                Repository.Release(Value);
                base.Dispose(disposing);
            }
        }

        private readonly ConcurrentQueue<T> _repo = new ConcurrentQueue<T>();

        public DisposableRepository(Func<T> create)
        {
            Create = create;
        }

        public Func<T> Create { get; }

        public Receipt Acquire()
        {
            T value;
            var found = _repo.TryDequeue(out value);
            return found ? new Receipt(this, value) : new Receipt(this, Create());
        }

        private void Release(T value)
        {
            _repo.Enqueue(value);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                foreach (var value in _repo)
                {
                    value.Dispose();
                }
            }
            base.Dispose(disposing);
        }
    }

    public class DisposableDictionary<TKey, TValue> : Disposable where TValue : IDisposable
    {
        private readonly ConcurrentDictionary<TKey, TValue> _dict = new ConcurrentDictionary<TKey, TValue>(); 

        public Func<TKey, TValue> Create { get; } 

        public DisposableDictionary(Func<TKey, TValue> create)
        {
            Create = create;
        }

        public TValue this[TKey key] => _dict.GetOrAdd(key, Create);

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                foreach (var value in _dict.Values)
                {
                    value.Dispose();
                }
                _dict.Clear();
            }
            base.Dispose(disposing);
        }
    }

    public class DeviceMemoryRepository : Disposable
    {
        public sealed class Receipt : Disposable
        {
            public Receipt(DeviceMemoryRepository repository, DeviceMemory<byte> memory)
            {
                Repository = repository;
                Memory = memory;
            }

            public DeviceMemoryRepository Repository { get; }

            public DeviceMemory<byte> Memory { get; }

            protected override void Dispose(bool disposing)
            {
                Repository.Release(Memory);
                base.Dispose(disposing);
            }
        }

        private readonly List<DeviceMemory<byte>> _repo = new List<DeviceMemory<byte>>();

        private class MemoryComparer : IComparer<DeviceMemory<Byte>>
        {
            public int Compare(DeviceMemory<byte> x, DeviceMemory<byte> y)
            {
                return x.Length.CompareTo(y.Length);
            }
        }

        static private readonly MemoryComparer MComp = new MemoryComparer();

        public DeviceMemoryRepository(Gpu gpu)
        {
            Gpu = gpu;
        }

        public Gpu Gpu { get; }

        public Receipt Acquire<T>(long length)
        {
            var size = Gpu.SizeOf<T>() * length;
            DeviceMemory<byte> memory;

            lock (_repo)
            {
                var idx = _repo.FindIndex(m => m.Size.ToInt64() >= size);
                if (idx >= 0)
                {
                    memory = _repo[idx];
                    _repo.RemoveAt(idx);
                    return new Receipt(this, memory);
                }
            }

            memory = Gpu.AllocateDevice<byte>(size);
            return new Receipt(this, memory);
        }

        private void Release(DeviceMemory<byte> memory)
        {
            lock (_repo)
            {
                _repo.Add(memory);
                _repo.Sort(MComp);
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                lock (_repo)
                {
                    foreach (var value in _repo)
                    {
                        value.Dispose();
                    }
                }
                _repo.Clear();
            }
            base.Dispose(disposing);
        }
    }

    public static class Util
    {
        public static string ToString<T>(IEnumerable<T> collection)
        {
            return $"[{string.Join(", ", collection.Select(elem => elem.ToString()))}]";
        }

        public static void EnsureEqual<T>(T expected, T actual, string message = null)
        {
            if (EqualityComparer<T>.Default.Equals(expected, actual)) return;
            if (message == null) throw new InvalidOperationException($"[EnsureEqual] Equality check fails: {actual} vs {expected}.");
            throw new InvalidOperationException($"[EnsureEqual] {message}: {actual} vs {expected}.");
        }

        public static void EnsureTrue(bool pred, string message = null)
        {
            if (pred) return;
            if (message == null) throw new InvalidOperationException("[EnsureTrue] Prediction is false.");
            throw new InvalidOperationException($"[EnsureTrue] {message}");
        }
    }

    internal static class UtilExtensions
    {
        public static void Iter<T>(this IEnumerable<T> ie, Action<T, int> action)
        {
            var i = 0;
            foreach (var e in ie)
            {
                action(e, i++);
            }
        }

        public static void LongIter<T>(this IEnumerable<T> ie, Action<T, long> action)
        {
            var i = 0L;
            foreach (var e in ie)
            {
                action(e, i++);
            }
        }

        public static int FirstIndex<T>(this IEnumerable<T> ie, Func<T, bool> pred)
        {
            var i = 0;
            foreach (var e in ie)
            {
                if (pred(e)) return i;
                i++;
            }
            return i;
        }

        public static void Dump(this TensorDescriptor desc)
        {
            DataType dataType;
            int nbDims;
            var dimA = new int[3];
            var strideA = new int[3];
            desc.GetND(out dataType, out nbDims, dimA, strideA);
            Console.WriteLine($"[TensorDescriptor] {nbDims} ({dimA[0]},{dimA[1]},{dimA[2]}) ({strideA[0]},{strideA[1]},{strideA[2]})");
        }
    }
}

