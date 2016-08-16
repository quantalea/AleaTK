using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Alea;

namespace AleaTK
{
    public class DimensionProperties<T> : IEnumerable<T>
    {
        private readonly T[] _properties;

        public DimensionProperties(T[] properties)
        {
            _properties = properties;
        }

        public int Rank => _properties.Length;

        public void EnsureDimension(int dimension)
        {
            if (dimension < 0 || dimension >= Rank)
            {
                throw new InvalidOperationException($"Wrong dimension {dimension} (Rank = {Rank}).");
            }
        }

        public T this[int dimension]
        {
            get
            {
                EnsureDimension(dimension);
                return _properties[dimension];
            }
        }

        public IEnumerator<T> GetEnumerator()
        {
            return _properties.Cast<T>().GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public override string ToString()
        {
            return Util.ToString(_properties);
        }

        public T[] AsArray => _properties;
    }

    public sealed class PartialShape : DimensionProperties<long>
    {
        public PartialShape(long[] properties) : base(properties)
        {
            var undeterminedDimensions = new List<int>();

            properties.Iter((size, dimension) =>
            {
                if (size < 0L)
                {
                    undeterminedDimensions.Add(dimension);
                }
            });

            UndeterminedDimensions = undeterminedDimensions.ToArray();
        }

        public int[] AsInt32Array => AsArray.Select(len => (int) len).ToArray();

        public int NumUndeterminedDimensions => UndeterminedDimensions.Length;

        public int[] UndeterminedDimensions { get; }

        public bool HasDeterminedDimension => NumUndeterminedDimensions < Rank;

        public bool HasUndeterminedDimension => NumUndeterminedDimensions > 0;

        public bool AllDimensionsDetermined => NumUndeterminedDimensions == 0;

        public long DeterminedLength => this.Where(l => l >= 0L).Aggregate(ScalarOps.Mul);

        public static PartialShape Create(params long[] lengths)
        {
            return new PartialShape(lengths);
        }

        public static PartialShape Reshape(PartialShape srcShape, PartialShape dstShape)
        {
            if (srcShape.AllDimensionsDetermined && dstShape.AllDimensionsDetermined)
            {
                // full shape, need check length
                Util.EnsureEqual(srcShape.DeterminedLength, dstShape.DeterminedLength);
                return dstShape;
            }

            // TODO: more check

            return dstShape;
        }

        public static PartialShape Broadcast(params PartialShape[] shapes)
        {
            // get the highest rank
            var rank = shapes.Select(shape => shape.Rank).Max();

            // extend shapes to that rank
            var extenededShapes =
                shapes.Select(shape => new PartialShape(Enumerable.Repeat(1L, rank - shape.Rank).Concat(shape).ToArray()))
                      .ToArray();

            // get the target length, also check if it breaks the rule
            var lengths = Enumerable.Range(0, rank).Select(dimension =>
            {
                var length = extenededShapes.Select(shape => shape[dimension]).Max();
                if (extenededShapes.Any(shape => shape[dimension] != -1 && shape[dimension] != 1L && shape[dimension] != length))
                {
                    var shapesStr = string.Join(",", shapes.Select(x => x.ToString()));
                    throw new InvalidOperationException($"Wrong shape operation, cannot broadcast. {shapesStr}");
                }
                return length;
            }).ToArray();

            return new PartialShape(lengths);
        }
    }

    public sealed class Shape : DimensionProperties<long>
    {
        public Shape(long[] lengths) : base(lengths)
        {
            if (lengths.Any(length => length <= 0))
            {
                throw new InvalidOperationException("Shape's length must have positive number.");
            }
        }

        public int[] AsInt32Array => AsArray.Select(len => (int)len).ToArray();

        public long Length => this.Aggregate(1L, ScalarOps.Mul);

        public IEnumerable<long> InnerChangeMostUnitStrides(long unitStride = 1L)
        {
            var sign = unitStride >= 0L ? 1L : -1L;
            return Enumerable.Range(0, Rank).Select(i => sign * this.Skip(i + 1).Aggregate(unitStride, ScalarOps.Mul));
        }

        public IEnumerable<long> OuterChangeMostUnitStrides(long unitStride = 1L)
        {
            var sign = unitStride >= 0L ? 1L : -1L;
            return Enumerable.Range(0, Rank).Select(i => sign * this.Take(i).Aggregate(unitStride, ScalarOps.Mul));
        }

        public static readonly Shape Scalar = Create();

        public static Shape Create(params long[] lengths)
        {
            return new Shape(lengths);
        }

        public static Shape GetArrayShape(Array array)
        {
            var lengths = Enumerable.Range(0, array.Rank).Select(array.GetLongLength).ToArray();
            return new Shape(lengths);
        }

        public static Shape Broadcast(params Shape[] shapes)
        {
            // get the highest rank
            var rank = shapes.Select(shape => shape.Rank).Max();

            // extend shapes to that rank
            var extenededShapes =
                shapes.Select(shape => new Shape(Enumerable.Repeat(1L, rank - shape.Rank).Concat(shape).ToArray()))
                      .ToArray();

            // get the target length, also check if it breaks the rule
            var lengths = Enumerable.Range(0, rank).Select(dimension =>
            {
                var length = extenededShapes.Select(shape => shape[dimension]).Max();
                if (extenededShapes.Any(shape => shape[dimension] != 1L && shape[dimension] != length))
                {
                    var shapesStr = string.Join(",", shapes.Select(x => x.ToString()));
                    throw new InvalidOperationException($"Wrong shape operation, cannot broadcast. {shapesStr}");
                }
                return length;
            }).ToArray();

            return new Shape(lengths);
        }

        public Shape Reshape(long[] dims)
        {
            var _dims = dims.Select(x => x).ToArray();

            // -1 means calc the shape, but only one -1 allowed.
            var numNegOne = _dims.Select(x => x < 0 ? 1 : 0).Sum();
            Util.EnsureTrue(numNegOne == 0 || numNegOne == 1);

            if (numNegOne == 0)
            {
                var shape = new Shape(_dims);
                // length must match old one
                Util.EnsureEqual(Length, shape.Length);
                return shape;
            }
            else
            {
                var remainLength = _dims.Select(x => x >= 0 ? x : 1L).Aggregate(ScalarOps.Mul);
                for (var i = 0; i < _dims.Length; ++i)
                {
                    if (_dims[i] < 0)
                    {
                        _dims[i] = Length / remainLength;
                        break;
                    }
                }
                // check if it is multiply correct
                var shape = new Shape(_dims);
                Util.EnsureEqual(Length, shape.Length);
                return shape;
            }
        }
    }

    public sealed class Strides : DimensionProperties<long>
    {
        public Strides(long[] strides) : base(strides)
        {
            DetectProperties();
        }

        public int[] AsInt32Array => AsArray.Select(len => (int)len).ToArray();

        private void DetectProperties()
        {
            if (Rank == 0)
            {
                IsInnerChangeMost = true;
                IsOuterChangeMost = true;
                return;
            }

            IsInnerChangeMost = false;
            IsOuterChangeMost = false;

            if (this.All(x => x >= 0L))
            {
                var order = this.ToList();

                order.Sort();
                if (order.SequenceEqual(this))
                {
                    IsOuterChangeMost = true;
                }

                order.Reverse();
                if (order.SequenceEqual(this))
                {
                    IsInnerChangeMost = true;
                }
            }
            else if (this.All(x => x <= 0L))
            {
                var order = this.ToList();

                order.Sort();
                if (order.SequenceEqual(this))
                {
                    IsInnerChangeMost = true;
                }

                order.Reverse();
                if (order.SequenceEqual(this))
                {
                    IsOuterChangeMost = true;
                }
            }
        }

        public bool IsInnerChangeMost { get; private set; }

        public bool IsOuterChangeMost { get; private set; }

        public static Strides Create(params long[] strides)
        {
            return new Strides(strides);
        }
    }

    public sealed class Layout
    {
        private long? _fullyUnitStride;

        private void DetectProperties()
        {
            var rank = Rank;
            var shape = Shape;
            var strides = Strides;

            // special handle for scalar
            if (rank == 0)
            {
                _fullyUnitStride = 0L;
                return;
            }

            _fullyUnitStride = null;

            if (strides.All(x => x > 0L))
            {
                var candidateFullyUnitStride = strides.Min();

                if (strides.SequenceEqual(shape.InnerChangeMostUnitStrides(candidateFullyUnitStride)))
                {
                    _fullyUnitStride = candidateFullyUnitStride;
                }
                else if (strides.SequenceEqual(shape.OuterChangeMostUnitStrides(candidateFullyUnitStride)))
                {
                    _fullyUnitStride = candidateFullyUnitStride;
                }
            }
            else if (strides.All(x => x == 0))
            {
                _fullyUnitStride = 0L;
            }
            else if (strides.All(x => x < 0L))
            {
                var candidateFullyUnitStride = strides.Max();

                if (strides.SequenceEqual(shape.InnerChangeMostUnitStrides(candidateFullyUnitStride)))
                {
                    _fullyUnitStride = candidateFullyUnitStride;
                }
                else if (strides.SequenceEqual(shape.OuterChangeMostUnitStrides(candidateFullyUnitStride)))
                {
                    _fullyUnitStride = candidateFullyUnitStride;
                }
            }
        }

        public Layout(Shape shape, Strides strides)
        {
            Util.EnsureEqual(shape.Rank, strides.Rank, "Shape and strides rank must equal");
            Shape = shape;
            Strides = strides;
            DetectProperties();
        }

        public Layout(Shape shape)
        {
            Shape = shape;
            Strides = new Strides(shape.InnerChangeMostUnitStrides().ToArray());
            DetectProperties();
        }

        public Shape Shape { get; }

        public Strides Strides { get; }

        public int Rank => Shape.Rank;

        public bool IsInnerChangeMost => Strides.IsInnerChangeMost;

        public bool IsOuterChangeMost => Strides.IsOuterChangeMost;

        public long FullyUnitStride
        {
            get
            {
                if (_fullyUnitStride != null) return _fullyUnitStride.Value;
                throw new InvalidOperationException("This layout is not fully unit stride.");
            }
        }

        public bool IsFullyUnitStride => _fullyUnitStride != null;

        public bool IsFullyUnitStrideOf(long stride)
        {
            return IsFullyUnitStride && FullyUnitStride == stride;
        }

        public bool IsFullyPacked => IsFullyUnitStrideOf(1L) || IsFullyUnitStrideOf(0L);

        public bool IsInnerChangeMostFullyPacked => IsInnerChangeMost && IsFullyPacked;

        public long PartialUnitStride1
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public bool IsPartialUnitStride1 => false;

        public bool IsPartialUnitStride1Of(long stride)
        {
            return false;
        }

        public bool IsPartialPacked1 => false;

        public IEnumerable<long> FlatIndices(int startDim = 0)
        {
            if (startDim == Shape.Rank)
            {
                yield return 0;
            }
            else
            {
                var length = Shape[startDim];
                var stride = Strides[startDim];

                for (var i = 0L; i < length; ++i)
                {
                    var offset = i * stride;
                    foreach (var x in FlatIndices(startDim + 1))
                    {
                        yield return x + offset;
                    }
                }
            }
        }

        public void Print<T>(Func<long, T> read, bool all = false)
        {
            Console.WriteLine("=====================================");
            Console.WriteLine($"Rank({Rank}) Shape({Shape}) Strides({Strides})");
            Console.WriteLine($"ICM({IsInnerChangeMost}) OCM({IsOuterChangeMost})");
            if (IsFullyPacked)
            {
                Console.WriteLine($"FullyUnitStride({FullyUnitStride})");
            }
            Console.WriteLine("-------------------------------------");

            if (Rank == 0)
            {
                Console.WriteLine($"Scalar: {read(0)}");
                Console.WriteLine("=====================================");
                return;
            }

            if (Rank == 1)
            {
                const long itemsPerRow = 5;
                var maxItems = all ? long.MaxValue : 100;
                var length = Shape[0];
                var stride = Strides[0];
                for (var i = 0L; i < length && i < maxItems; ++i)
                {
                    if (i % itemsPerRow == 0)
                    {
                        Console.Write($"#.{i:D4}: \t");
                    }
                    Console.Write($"{read(stride*i)}");
                    if (i % itemsPerRow == itemsPerRow - 1)
                    {
                        Console.WriteLine();
                    }
                    else if (i == maxItems - 1)
                    {
                        Console.WriteLine();
                    }
                    else
                    {
                        Console.Write("\t");
                    }
                }
                if (Shape.Length > maxItems)
                {
                    Console.WriteLine("...");
                }
                else
                {
                    Console.WriteLine();
                }
                Console.WriteLine("=====================================");
                return;
            }

            if (Rank == 2)
            {
                var maxCols = all ? long.MaxValue : 10;
                var maxRows = all ? long.MaxValue : 10;
                var rows = Shape[0];
                var cols = Shape[1];
                var rowStride = Strides[0];
                var colStride = Strides[1];
                for (var row = 0L; row < rows && row < maxRows; ++row)
                {
                    Console.Write($"#.{row:D4}: \t");
                    for (var col = 0L; col < cols && col < maxCols; ++col)
                    {
                        if (col != 0L) Console.Write("\t");
                        Console.Write($"{read(row*rowStride + col*colStride)}");
                    }
                    if (maxCols < cols) Console.WriteLine("  ...");
                    else Console.WriteLine();
                }
                if (maxRows < rows) Console.WriteLine("...");
                Console.WriteLine("=====================================");
                return;
            }

            throw new NotImplementedException();
        }

        public static bool CanFullyUnitStrideMapping(params Layout[] layouts)
        {
            // null layout is not allowed
            if (layouts.Any(layout => layout == null)) return false;

            // all layout need to be fully unit stride
            if (layouts.Any(layout => !layout.IsFullyUnitStride)) return false;

            // the direction should be same
            if (layouts.All(layout => layout.IsInnerChangeMost)) return true;

            if (layouts.All(layout => layout.IsOuterChangeMost)) return true;

            return false;
        }

        public static bool Match(Layout l1, Layout l2)
        {
            Util.EnsureTrue(l1 != null);
            Util.EnsureTrue(l2 != null);
            if (l1 == null || l2 == null) return false;
            if (!l1.Shape.SequenceEqual(l2.Shape)) return false;
            if (!l1.Strides.SequenceEqual(l2.Strides)) return false;
            return true;
        }
    }
}