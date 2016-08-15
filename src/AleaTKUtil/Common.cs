using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using NUnit.Framework;

namespace AleaTKUtil
{
    public static class Common
    {
        public static Random Rng = new Random(42);

        public static double[,] Dot(double[,] matA, double[,] matB)
        {
            var aRows = matA.GetLength(0);
            var aCols = matA.GetLength(1);
            var bRows = matB.GetLength(0);
            var bCols = matB.GetLength(1);

            if (aCols != bRows)
            {
                throw new Exception("Wrong size");
            }

            var matC = new double[aRows, bCols];
            for (var i = 0; i < aRows; ++i)
            {
                for (var j = 0; j < bCols; ++j)
                {
                    for (var k = 0; k < aCols; ++k)
                    {
                        matC[i, j] += matA[i, k] * matB[k, j];
                    }
                }
            }
            return matC;
        }

        public static double[] Dot(double[,] mat, double[] vec)
        {
            var aRows = mat.GetLength(0);
            var aCols = mat.GetLength(1);

            if (aCols != vec.GetLength(0))
            {
                throw new Exception("Wrong size");
            }

            var prod = new double[aRows];
            for (var i = 0; i < aRows; ++i)
            {
                prod[i] = 0;
                for (var j = 0; j < aCols; ++j)
                {
                    prod[i] += mat[i, j]*vec[j];
                }
            }
            return prod;
        }

        public static double[] Dot(double[] vec, double[,] mat)
        {
            var aRows = mat.GetLength(0);
            var aCols = mat.GetLength(1);

            if (aRows != vec.GetLength(0))
            {
                throw new Exception("Wrong size");
            }

            var prod = new double[aCols];
            for (var i = 0; i < aCols; ++i)
            {
                prod[i] = 0;
                for (var j = 0; j < aRows; ++j)
                {
                    prod[i] += vec[j]*mat[j, i];
                }
            }
            return prod;
        }

        public static float[,] Dot(float[,] matA, float[,] matB)
        {
            var aRows = matA.GetLength(0);
            var aCols = matA.GetLength(1);
            var bRows = matB.GetLength(0);
            var bCols = matB.GetLength(1);

            if (aCols != bRows)
            {
                throw new Exception("Wrong size");
            }

            var matC = new float[aRows, bCols];
            for (var i = 0; i < aRows; ++i)
            {
                for (var j = 0; j < bCols; ++j)
                {
                    for (var k = 0; k < aCols; ++k)
                    {
                        matC[i, j] += matA[i, k]*matB[k, j];
                    }
                }
            }
            return matC;
        }

        public static float[] Dot(float[,] mat, float[] vec)
        {
            var aRows = mat.GetLength(0);
            var aCols = mat.GetLength(1);

            if (aCols != vec.GetLength(0))
            {
                throw new Exception("Wrong size");
            }

            var prod = new float[aRows];
            for (var i = 0; i < aRows; ++i)
            {
                prod[i] = 0;
                for (var j = 0; j < aCols; ++j)
                {
                    prod[i] += mat[i, j]*vec[j];
                }
            }
            return prod;
        }

        public static float[] Dot(float[] vec, float[,] mat)
        {
            var aRows = mat.GetLength(0);
            var aCols = mat.GetLength(1);

            if (aRows != vec.GetLength(0))
            {
                throw new Exception("Wrong size");
            }

            var prod = new float[aCols];
            for (var i = 0; i < aCols; ++i)
            {
                prod[i] = 0;
                for (var j = 0; j < aRows; ++j)
                {
                    prod[i] += vec[j]*mat[j, i];
                }
            }
            return prod;
        }

        public static double[,] Mul(this double[,] matA, double scalar)
        {
            var rows = matA.GetLength(0);
            var cols = matA.GetLength(1);

            var matR = new double[rows, cols];
            for (var row = 0; row < rows; ++row)
            {
                for (var col = 0; col < cols; ++col)
                {
                    matR[row, col] = matA[row, col]*scalar;
                }
            }
            return matR;
        }

        public static double[,] Add(this double[,] matA, double[,] matB)
        {
            var rows = matA.GetLength(0);
            var cols = matA.GetLength(1);

            if (matB.GetLength(0) != rows || matB.GetLength(1) != cols)
            {
                throw new Exception("Bad size");
            }

            var matR = new double[rows, cols];
            for (var row = 0; row < rows; ++row)
            {
                for (var col = 0; col < cols; ++col)
                {
                    matR[row, col] = matA[row, col] + matB[row, col];
                }
            }
            return matR;
        }

        public static T[,] Transpose<T>(T[,] input)
        {
            var rows = input.GetLength(0);
            var cols = input.GetLength(1);
            var output = new T[cols, rows];
            for (var i = 0; i < rows; ++i)
            {
                for (var j = 0; j < cols; ++j)
                {
                    output[j, i] = input[i, j];
                }
            }
            return output;
        }

        public static Random Random = new Random();

        public static T[] GenerateRandomData<T>(int numElements)
        {
            var rawData = new byte[numElements * Marshal.SizeOf(typeof(T))];
            Random.NextBytes(rawData);
            var data = new T[numElements];
            Buffer.BlockCopy(rawData, 0, data, 0, rawData.Length);
            return data;
        }

        public static bool[] GenerateRandomBooleanData(int numElements)
        {
            return Enumerable.Range(0, numElements).Select(_ => Random.Next(0, 2) == 1).ToArray();
        }

        public static double[] GenerateRandomDoubleData(int numElements, double minValue, double maxValue)
        {
            return Enumerable.Range(0, numElements).Select(_ => Random.NextDouble() * (maxValue - minValue) + minValue).ToArray();
        }

        public static float[] GenerateRandomSingleData(int numElements, double minValue, double maxValue)
        {
            return Enumerable.Range(0, numElements).Select(_ => (float)(Random.NextDouble() * (maxValue - minValue) + minValue)).ToArray();
        }

        public static void Dump<T>(T[] array)
        {
            for (var i = 0; i < array.Length; ++i)
            {
                Console.WriteLine($"#.{i}: {array[i]}");
            }
        }

        public static void Dump<T1, T2>(T1[] array1, T2[] array2)
        {
            Assert.AreEqual(array1.Length, array2.Length);
            for (var i = 0; i < array1.Length; ++i)
            {
                Console.WriteLine($"#.{i}: {array1[i]} - {array2[i]}");
            }
        }

        public static void Dump<T1, T2>(T1[] array1, T2[] array2, int atMost)
        {
            Assert.AreEqual(array1.Length, array2.Length);
            for (var i = 0; i < array1.Length && i < atMost; ++i)
            {
                Console.WriteLine($"#.{i}: {array1[i]} - {array2[i]}");
            }
        }

        public static void Iter<T>(this IEnumerable<T> ie, Action<T, int> action)
        {
            var i = 0;
            foreach (var e in ie)
            {
                action(e, i++);
            }
        }

        public static Func<int, double> UniformRandomDouble(Random rng)
        {
            return i => rng.NextDouble();
        }

        public static Func<int, float> UniformRandomSingle(Random rng)
        {
            return i => (float)rng.NextDouble();
        }

        public static Func<int, int> UniformRandomInt(Random rng)
        {
            return i => rng.Next();
        }

        public static T[] CreateArray<T>(int n, Func<int, T> init)
        {
            var array = new T[n];
            for (var i = 0; i < array.GetLength(0); ++i)
            {
                array[i] = init(i);
            }
            return array;
        }

        public static void InitArray<T>(T[] array, Func<int, T> init)
        {
            for (var i = 0; i < array.GetLength(0); ++i)
            {
                array[i] = init(i);
            }
        }

        public static void UniformRandomArray(double[] array, Random rng = null)
        {
            var gen = rng ?? Rng;
            InitArray(array, i => gen.NextDouble());
        }

        public static void UniformRandomArray(float[] array, Random rng = null)
        {
            var gen = rng ?? Rng;
            InitArray(array, i => (float)gen.NextDouble());
        }

        public static void UniformRandomArray(int[] array, Random rng = null)
        {
            var gen = rng ?? Rng;
            InitArray(array, i => gen.Next());
        }

        public static void NormalRandomArray(double[] array, Random rng = null)
        {
            var gen = rng ?? Rng;
            InitArray(array, i =>
            {
                var u1 = gen.NextDouble();
                var u2 = gen.NextDouble();
                var normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                return normal;
            });
        }

        public static void NormalRandomArray(float[] array, Random rng = null)
        {
            var gen = rng ?? Rng;
            InitArray(array, i =>
            {
                var u1 = gen.NextDouble();
                var u2 = gen.NextDouble();
                var normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                return (float)normal;
            });
        }

        public static T[,] CreateArray<T>(int n1, int n2, Func<int, int, T> init)
        {
            var array = new T[n1, n2];
            for (var i = 0; i < array.GetLength(0); ++i)
            {
                for (var j = 0; j < array.GetLength(1); ++j)
                {
                    array[i, j] = init(i, j);
                }
            }
            return array;
        }

        public static void InitArray<T>(T[,] array, Func<int, int, T> init)
        {
            for (var i = 0; i < array.GetLength(0); ++i)
            {
                for (var j = 0; j < array.GetLength(1); ++j)
                {
                    array[i, j] = init(i, j);
                }
            }
        }

        public static void UniformRandomArray(double[,] array, Random rng = null)
        {
            var gen = rng ?? Rng;
            InitArray(array, (i, j) => gen.NextDouble());
        }

        public static void UniformRandomArray(float[,] array, Random rng = null)
        {
            var gen = rng ?? Rng;
            InitArray(array, (i, j) => (float)gen.NextDouble());
        }

        public static void UniformRandomArray(int[,] array, Random rng = null)
        {
            var gen = rng ?? Rng;
            InitArray(array, (i, j) => gen.Next());
        }

        public static void NormalRandomArray(double[,] array, Random rng = null)
        {
            var gen = rng ?? Rng;
            InitArray(array, (i, j) =>
            {
                var u1 = gen.NextDouble();
                var u2 = gen.NextDouble();
                var normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                return normal;
            });
        }

        public static void NormalRandomArray(float[,] array, Random rng = null)
        {
            var gen = rng ?? Rng;
            InitArray(array, (i, j) =>
            {
                var u1 = gen.NextDouble();
                var u2 = gen.NextDouble();
                var normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                return (float)normal;
            });
        }

        public static T[,,] CreateArray<T>(int n1, int n2, int n3, Func<int, int, int, T> init)
        {
            var array = new T[n1, n2, n3];
            for (var i = 0; i < array.GetLength(0); ++i)
            {
                for (var j = 0; j < array.GetLength(1); ++j)
                {
                    for (var k = 0; k < array.GetLength(2); ++k)
                    {
                        array[i, j, k] = init(i, j, k);
                    }
                }
            }
            return array;
        }

        public static void InitArray<T>(T[,,] array, Func<int, int, int, T> init)
        {
            for (var i = 0; i < array.GetLength(0); ++i)
            {
                for (var j = 0; j < array.GetLength(1); ++j)
                {
                    for (var k = 0; k < array.GetLength(2); ++k)
                    {
                        array[i, j, k] = init(i, j, k);
                    }
                }
            }
        }

        public static void UniformRandomArray(double[,,] array, Random rng = null)
        {
            var gen = rng ?? Rng;
            InitArray(array, (i, j, k) => gen.NextDouble());
        }

        public static void UniformRandomArray(float[,,] array, Random rng = null)
        {
            var gen = rng ?? Rng;
            InitArray(array, (i, j, k) => (float)gen.NextDouble());
        }

        public static void UniformRandomArray(int[,,] array, Random rng = null)
        {
            var gen = rng ?? Rng;
            InitArray(array, (i, j, k) => gen.Next());
        }

        public static void NormalRandomArray(double[,,] array, Random rng= null)
        {
            var gen = rng ?? Rng;
            InitArray(array, (i, j, k) =>
            {
                var u1 = gen.NextDouble();
                var u2 = gen.NextDouble();
                var normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                return normal;
            });
        }

        public static void NormalRandomArray(float[,,] array, Random rng = null)
        {
            var gen = rng ?? Rng;
            InitArray(array, (i, j, k) =>
            {
                var u1 = gen.NextDouble();
                var u2 = gen.NextDouble();
                var normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                return (float)normal;
            });
        }

        public static void AreClose(double[] expected, double[] actual, double error)
        {
            if (expected.Length != actual.Length)
            {
                Assert.Fail($"Length doesn't match: {expected.Length} vs {actual.Length}");
            }

            for (var i = 0; i < expected.Length; ++i)
            {
                Assert.That(actual[i], Is.EqualTo(expected[i]).Within(error));
            }
        }

        public static double MaxAbsDiff(float[] expected, float[] actual)
        {
            return (double)expected.Zip(actual, (e, a) => Math.Abs(a - e)).Max();
        }

        public static double MaxAbsDiff(double[] expected, double[] actual)
        {
            return expected.Zip(actual, (e, a) => Math.Abs(a - e)).Max();
        }

        public static void AreClose(float[] expected, float[] actual, double error)
        {
            if (expected.Length != actual.Length)
            {
                Assert.Fail($"Length doesn't match: {expected.Length} vs {actual.Length}");
            }

            for (var i = 0; i < expected.Length; ++i)
            {
                Assert.That(actual[i], Is.EqualTo(expected[i]).Within(error));
            }
        }

        public static void AreEqual<T>(T[,] expected, T[,] actual)
        {
            Assert.AreEqual(expected.GetLength(0), actual.GetLength(0));
            Assert.AreEqual(expected.GetLength(1), actual.GetLength(1));
            for (var i = 0; i < expected.GetLength(0); ++i)
            {
                for (var j = 0; j < expected.GetLength(1); ++j)
                {
                    Assert.AreEqual(expected[i, j], actual[i, j]);
                }
            }
        }

        public static void AreClose(double[,] expected, double[,] actual, double error)
        {
            Assert.AreEqual(expected.GetLength(0), actual.GetLength(0));
            Assert.AreEqual(expected.GetLength(1), actual.GetLength(1));
            for (var i = 0; i < expected.GetLength(0); ++i)
            {
                for (var j = 0; j < expected.GetLength(1); ++j)
                {
                    Assert.That(actual[i, j], Is.EqualTo(expected[i, j]).Within(error));
                }
            }
        }

        public static void AreClose(double[,,] expected, double[,,] actual, double error)
        {
            Assert.AreEqual(expected.GetLength(0), actual.GetLength(0));
            Assert.AreEqual(expected.GetLength(1), actual.GetLength(1));
            Assert.AreEqual(expected.GetLength(2), actual.GetLength(2));
            for (var i = 0; i < expected.GetLength(0); ++i)
            {
                for (var j = 0; j < expected.GetLength(1); ++j)
                {
                    for (var k = 0; k < expected.GetLength(2); ++k)
                    {
                        Assert.That(actual[i, j, k], Is.EqualTo(expected[i, j, k]).Within(error));
                    }
                }
            }
        }

        public static void AreClose(float[,] expected, float[,] actual, double error)
        {
            Assert.AreEqual(expected.GetLength(0), actual.GetLength(0));
            Assert.AreEqual(expected.GetLength(1), actual.GetLength(1));
            for (var i = 0; i < expected.GetLength(0); ++i)
            {
                for (var j = 0; j < expected.GetLength(1); ++j)
                {
                    Assert.That(actual[i, j], Is.EqualTo(expected[i, j]).Within(error));
                }
            }
        }

        public static void AreClose(float[,,] expected, float[,,] actual, double error)
        {
            Assert.AreEqual(expected.GetLength(0), actual.GetLength(0));
            Assert.AreEqual(expected.GetLength(1), actual.GetLength(1));
            Assert.AreEqual(expected.GetLength(2), actual.GetLength(2));
            for (var i = 0; i < expected.GetLength(0); ++i)
            {
                for (var j = 0; j < expected.GetLength(1); ++j)
                {
                    for (var k = 0; k < expected.GetLength(2); ++k)
                    {
                        Assert.That(actual[i, j, k], Is.EqualTo(expected[i, j, k]).Within(error));
                    }
                }
            }
        }
    }
}
